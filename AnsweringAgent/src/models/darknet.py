import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
from utils.logger import get_logger
from config import Config
logger = get_logger()

def create_modules(module_defs: List[Dict[str, Any]]) -> Tuple[nn.ModuleList, List[int]]:
    """
    Constructs module list of layer blocks from module configuration in module_defs.
    
    Args:
        module_defs: List of module definitions from config file
        
    Returns:
        Tuple containing:
            - ModuleList of constructed layers
            - List of output filter counts for each layer
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        
        if module_def['type'] == 'convolutional':
            modules.add_module(f'conv_{i}', _create_conv_layer(module_def, output_filters[-1]))
            filters = int(module_def['filters'])
            
        elif module_def['type'] == 'upsample':
            modules.add_module(f'upsample_{i}', nn.Upsample(scale_factor=int(module_def['stride'])))
            filters = output_filters[-1]
            
        elif module_def['type'] == 'route':
            modules.add_module(f'route_{i}', EmptyLayer())
            layers = [int(x) for x in module_def["layers"].split(',')]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            
        elif module_def['type'] == 'shortcut':
            modules.add_module(f'shortcut_{i}', EmptyLayer())
            filters = output_filters[-1]
            
        elif module_def['type'] == 'yolo':
            yolo_layer = _create_yolo_layer(module_def, hyperparams)
            modules.add_module(f'yolo_{i}', yolo_layer)
            filters = output_filters[-1]
            
        module_list.append(modules)
        output_filters.append(filters)
        
    return module_list, output_filters

def _create_conv_layer(module_def: Dict[str, Any], in_channels: int) -> nn.Sequential:
    """Create a convolutional layer with optional batch normalization and activation."""
    bn = int(module_def['batch_normalize'])
    filters = int(module_def['filters'])
    kernel_size = int(module_def['size'])
    pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
    
    modules = nn.Sequential()
    modules.add_module('conv', nn.Conv2d(
        in_channels=in_channels,
        out_channels=filters,
        kernel_size=kernel_size,
        stride=int(module_def['stride']),
        padding=pad,
        bias=not bn
    ))
    
    if bn:
        modules.add_module('batch_norm', nn.BatchNorm2d(filters))
    if module_def['activation'] == 'leaky':
        modules.add_module('leaky', nn.LeakyReLU(0.1))
        
    return modules

def _create_yolo_layer(module_def: Dict[str, Any], hyperparams: Dict[str, Any]) -> 'YOLOLayer':
    """Create a YOLO detection layer."""
    anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
    anchors = [float(x) for x in module_def["anchors"].split(",")]
    anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
    anchors = [anchors[i] for i in anchor_idxs]
    
    return YOLOLayer(
        anchors=anchors,
        num_classes=int(module_def['classes']),
        img_dim=int(hyperparams['height']),
        anchor_idxs=anchor_idxs
    )

class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers."""
    def __init__(self):
        super(EmptyLayer, self).__init__()

class YOLOLayer(nn.Module):
    """YOLO detection layer."""
    
    def __init__(self, anchors: List[tuple], num_classes: int, img_dim: int, anchor_idxs: List[int]):
        """
        Initialize YOLO layer.
        
        Args:
            anchors: List of anchor box dimensions
            num_classes: Number of object classes
            img_dim: Input image dimension
            anchor_idxs: Indices of anchors to use
        """
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.anchor_idxs = anchor_idxs
        self.num_anchors = len(anchors)
        self.bbox_attrs = 5 + num_classes
        
        # Build anchor grids
        stride = self._get_stride()
        nG = int(self.img_dim / stride)
        self._init_grids(nG)
        
    def _get_stride(self) -> int:
        """Determine stride based on anchor indices."""
        if self.anchor_idxs[0] == (self.num_anchors * 2):
            return 32
        elif self.anchor_idxs[0] == self.num_anchors:
            return 16
        return 8
        
    def _init_grids(self, nG: int):
        """Initialize grid and anchor tensors."""
        self.grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).float()
        self.grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).float()
        self.scaled_anchors = torch.FloatTensor([(a_w / self._get_stride(), a_h / self._get_stride()) 
                                               for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of YOLO layer.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Detection predictions
        """
        batch_size = x.size(0)
        grid_size = x.size(2)
        stride = self.img_dim / grid_size
        
        # Reshape prediction
        prediction = x.view(batch_size, self.num_anchors, self.bbox_attrs, grid_size, grid_size)
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
        
        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.
        
        # Move grid and anchors to device
        grid_x = self.grid_x.to(x.device).repeat(batch_size, self.num_anchors, 1, 1)
        grid_y = self.grid_y.to(x.device).repeat(batch_size, self.num_anchors, 1, 1)
        scaled_anchors = self.scaled_anchors.to(x.device).repeat(batch_size, 1, 1, 1)
        
        # Add offset and scale with anchors
        pred_boxes = torch.zeros(prediction[..., :4].shape, device=x.device)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2:4] = torch.exp(w.data) * scaled_anchors
        
        # Reshape to [batch_size, num_anchors * grid_size * grid_size, bbox_attrs]
        return torch.cat(
            (pred_boxes.view(batch_size, -1, 4),
             pred_conf.view(batch_size, -1, 1),
             pred_cls.view(batch_size, -1, self.num_classes)),
            -1
        )

class Darknet(nn.Module):
    """Darknet backbone for feature extraction."""
    
    def __init__(self, config: Config):
        """
        Initialize Darknet model.
        
        Args:
            config: Configuration object
        """
        super(Darknet, self).__init__()
        self.device = torch.device(config.training.device)
        logger.info(f"Initializing Darknet on device: {self.device}")
        
        self.module_defs = self._parse_model_config(config.model.config_path)
        self.module_defs[0]['height'] = config.model.img_size
        self.module_list, self.output_filters = create_modules(self.module_defs)
        self.img_size = config.model.img_size
        
        # Move to appropriate device
        self.to(self.device)
        
    def _parse_model_config(self, path: str) -> List[Dict[str, Any]]:
        """
        Parse the YOLO layer configuration file.
        
        Args:
            path: Path to config file
            
        Returns:
            List of module definitions
        """
        with open(path, 'r') as f:
            lines = [line.strip() for line in f.read().split('\n') 
                    if line.strip() and not line.startswith('#')]
            
        module_defs = []
        for line in lines:
            if line.startswith('['):
                module_defs.append({'type': line[1:-1].rstrip()})
                if module_defs[-1]['type'] == 'convolutional':
                    module_defs[-1]['batch_normalize'] = 0
            else:
                key, value = line.split("=")
                module_defs[-1][key.rstrip()] = value.strip()
                
        return module_defs
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Model output
        """
        img_dim = x.size(2)
        layer_outputs = []
        yolo_outputs = []
        
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample', 'yolo']:
                x = module(x)
            elif module_def['type'] == 'route':
                layer_i = [int(x) for x in module_def["layers"].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i if i < len(layer_outputs)], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                if layer_i < len(layer_outputs):
                    x = layer_outputs[-1] + layer_outputs[layer_i]
                    
            layer_outputs.append(x)
            if module_def['type'] == 'yolo':
                yolo_outputs.append(x)
                
        return yolo_outputs[-1] if yolo_outputs else layer_outputs[-1]
        
    def to_device(self, device: str):
        """Move the model and all its parameters to the specified device."""
        self.device = torch.device(device)
        logger.debug(f"Moving Darknet to device: {self.device}")
        
        # Move all modules to device
        for i, module in enumerate(self.module_list):
            module.to(self.device)
            logger.debug(f"Module {i} moved to device: {self.device}")
            
            if isinstance(module, nn.Sequential):
                for j, layer in enumerate(module):
                    layer.to(self.device)
                    if isinstance(layer, nn.Conv2d):
                        logger.debug(f"Module {i}, Layer {j} (Conv2d) weights device: {layer.weight.device}")
                    elif isinstance(layer, nn.BatchNorm2d):
                        logger.debug(f"Module {i}, Layer {j} (BatchNorm2d) weights device: {layer.weight.device}")
                    elif isinstance(layer, YOLOLayer):
                        layer.grid_x = layer.grid_x.to(self.device)
                        layer.grid_y = layer.grid_y.to(self.device)
                        layer.scaled_anchors = layer.scaled_anchors.to(self.device)
                        layer.anchor_w = layer.anchor_w.to(self.device)
                        layer.anchor_h = layer.anchor_h.to(self.device)
                        logger.debug(f"Module {i}, Layer {j} (YOLOLayer) moved to device: {self.device}")
                        
        # Verify all components are on the correct device
        self._verify_device_consistency()
        
    def _verify_device_consistency(self):
        """Verify that all model components are on the correct device."""
        for name, param in self.named_parameters():
            if param.device != self.device:
                logger.warning(f"Parameter {name} is on {param.device} instead of {self.device}")
                param.data = param.data.to(self.device)
                
        # Verify all modules
        for i, module in enumerate(self.module_list):
            if isinstance(module, nn.Sequential):
                for j, layer in enumerate(module):
                    if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)):
                        if layer.weight.device != self.device:
                            logger.warning(f"Module {i}, Layer {j} weights are on {layer.weight.device} instead of {self.device}")
                            layer.to(self.device)
                    elif isinstance(layer, YOLOLayer):
                        if layer.grid_x.device != self.device:
                            logger.warning(f"Module {i}, Layer {j} (YOLOLayer) tensors are on {layer.grid_x.device} instead of {self.device}")
                            layer.grid_x = layer.grid_x.to(self.device)
                            layer.grid_y = layer.grid_y.to(self.device)
                            layer.scaled_anchors = layer.scaled_anchors.to(self.device)
                            layer.anchor_w = layer.anchor_w.to(self.device)
                            layer.anchor_h = layer.anchor_h.to(self.device) 