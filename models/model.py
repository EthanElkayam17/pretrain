import yaml
import os
import logging
import torch
import torch.nn as nn
import math
from more_itertools import unique_everseen
from functools import partial
from typing import Iterator, List, Tuple, Union
from torch.nn.parameter import Parameter
from torchvision.ops import StochasticDepth, SqueezeExcitation, Conv2dNormActivation
from utils.other import dirjoin

CFG_PATH = 'configs/architecture'


class InvertedResidualDWBlock(nn.Module):
    """Inverted Residual Depthwise Block based on:
    "MobileNetV2: Inverted Residuals and Linear Bottlenecks" - https://arxiv.org/pdf/1801.04381
    with squeeze-and-excitation and stochastic depth.
    """
    
    def __init__(
            self,
            stride: int,
            kernel_size: int,
            in_channels: int,
            out_channels: int,
            expansion_ratio: float,
            stochastic_depth_prob: float,
            SE_reduction_ratio: float = 0.25,
            fused: bool = False
    ) -> None:
        """Constructs an Inverted Residual Block
        
        Args:
            stride: stride of depthwise convolution
            kernel_size: kernel size of depthwise convolution
            in_channels: number of channels in input map
            out_channels: desired number of channels in output map
            expansion_ratio: multiplier to expand in_channels by
            stochastic_depth_prob: probability for dropout in stochastic depth
            SE_reduction_ratio: squeeze ratio for squeeze-and-excitation
            fused: is DWconv switched to regular conv
        
        Note: block will be residual iff stride==1 && in_channels==out_channels,
        otherwise block will be a standard inverted depthwise bottleneck"""
        
        super().__init__()
        self.fused = fused
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stochastic_depth_prob = stochastic_depth_prob
        self.is_residual = stride==1 and in_channels==out_channels

        activation_layer = nn.SiLU
        norm_layer = partial(nn.BatchNorm2d,eps=0.003,momentum=0.925)

        layers: List[nn.Module] = []
        expanded_channels = int(round(in_channels*expansion_ratio))

        if not self.fused:
            #expand
            if expanded_channels != in_channels:
                layers.append(
                    Conv2dNormActivation(in_channels=self.in_channels,
                                        out_channels=expanded_channels,
                                        kernel_size=1,
                                        norm_layer=norm_layer,
                                        activation_layer=activation_layer
                                        )
                )
        
            #DWconv
            layers.append(
                Conv2dNormActivation(in_channels=expanded_channels,
                                    out_channels=expanded_channels,
                                    kernel_size=self.kernel_size,
                                    stride=self.stride,
                                    padding=(math.ceil((self.kernel_size - self.stride)/2)),
                                    groups=expanded_channels, # -> depthwise
                                    norm_layer=norm_layer,
                                    activation_layer=activation_layer
                                    )
            )
        else:
            layers.append(
                Conv2dNormActivation(in_channels=self.in_channels,
                                    out_channels=expanded_channels,
                                    kernel_size=self.kernel_size,
                                    stride=self.stride,
                                    norm_layer=norm_layer,
                                    activation_layer=activation_layer)
            )

        #squeeze-and-excitation
        if SE_reduction_ratio != 0:
            squeeze_channels = max(1,int(round(in_channels*SE_reduction_ratio)))
            layers.append(
                SqueezeExcitation(expanded_channels, squeeze_channels, activation=partial(nn.SiLU,inplace=True))
            )

        if expanded_channels != out_channels:
            #shrink
            layers.append(
                Conv2dNormActivation(in_channels=expanded_channels,
                                    out_channels=self.out_channels,
                                    kernel_size=1,
                                    norm_layer=norm_layer,
                                    activation_layer=None)
            )

        self.block = nn.Sequential(*layers)

        self.stochastic_depth = StochasticDepth(self.stochastic_depth_prob, "row")
    
    def forward(self, x):
        y = self.block(x)

        if self.is_residual:
            return self.stochastic_depth(y) + x
        return y


class CFGCNN(nn.Module):
    """CNN based on config file"""
    
    def __init__(
            self,
            cfg_name: str,
            cfg_dir: str = CFG_PATH,
            stochastic_depth_incremental: float = 0.004,
            dropout_prob_override: float = -1,
            logger = None) -> None:
        """Constructs CNN based on config file
        
        Args:
            cfg_name: name of config file.
            cfg_dir: path to config directory.
            stochastic_depth_incremental: step size for stochastic depth for each layer.
            dropout_prob_override: override dropout_prob on config file.
            logger: logging function.
        """
        
        super().__init__()
        self.out_features = 0
        self.layer_names = []
        
        layers: List[nn.Module] = []
        classifier_layers: List[nn.Module] = []
        self.inner_stages: List[nn.Module] = []
        
        raw_param_names = []
        current_stochastic_depth = 0

        
        if logger is None:
            logger = logging.getLogger('null_logger')
            logger.addHandler(logging.NullHandler)

        logger.info("Creating CFGCNN")
        cfg_path = (dirjoin(cfg_dir,cfg_name))
        with open(cfg_path, 'r') as cfg_file:
            cfg = (yaml.safe_load(cfg_file))
        

        for stage in cfg.get('model_stages'):
            logger.info(f"Adding {stage.get('type')} stage")
            match stage.get('type'):
                case 'conv':
                    layers.append(
                        Conv2dNormActivation(kernel_size=stage.get('kernel'),
                                            stride=stage.get('stride'),
                                            in_channels=stage.get('in_channels'),
                                            out_channels=stage.get('out_channels'),
                                            norm_layer=partial(nn.BatchNorm2d,eps=0.003,momentum=0.925),
                                            activation_layer=nn.SiLU)
                    )
                    

                    for _ in range(stage.get('layers') - 1):
                        layers.append(
                            Conv2dNormActivation(stride=1, #all stacked layers in the same stage gets stride=1
                                                kernel_size=stage.get('kernel'),
                                                in_channels=stage.get('out_channels'),
                                                out_channels=stage.get('out_channels'))
                        )
                    
                case 'fused':
                    layers.append(
                        InvertedResidualDWBlock(stride=stage.get('stride'),                                                
                                            kernel_size=stage.get('kernel'),
                                            in_channels=stage.get('in_channels'),
                                            out_channels=stage.get('out_channels'),
                                            expansion_ratio=stage.get('expansion_ratio'),
                                            stochastic_depth_prob=current_stochastic_depth,
                                            SE_reduction_ratio=stage.get('SE_ratio'),
                                            fused=True)
                    )
                    current_stochastic_depth += stochastic_depth_incremental

                    for _ in range(stage.get('layers') - 1):
                        layers.append(
                            InvertedResidualDWBlock(stride=1,
                                                kernel_size=stage.get('kernel'),
                                                in_channels=stage.get('out_channels'),                                                    
                                                out_channels=stage.get('out_channels'),
                                                expansion_ratio=stage.get('expansion_ratio'),
                                                stochastic_depth_prob=current_stochastic_depth,
                                                SE_reduction_ratio=stage.get('SE_ratio'),
                                                fused=True)
                            )
                        current_stochastic_depth += stochastic_depth_incremental
                    
                case 'invres':
                    layers.append(
                        InvertedResidualDWBlock(stride=stage.get('stride'),
                                            kernel_size=stage.get('kernel'),
                                            in_channels=stage.get('in_channels'),
                                            out_channels=stage.get('out_channels'),
                                            expansion_ratio=stage.get('expansion_ratio'),
                                            stochastic_depth_prob=current_stochastic_depth,
                                            SE_reduction_ratio=stage.get('SE_ratio'))
                    )
                    current_stochastic_depth += stochastic_depth_incremental

                    for _ in range(stage.get('layers') - 1):
                        layers.append(
                            InvertedResidualDWBlock(stride=1,
                                                kernel_size=stage.get('kernel'),
                                                in_channels=stage.get('out_channels'),
                                                out_channels=stage.get('out_channels'),
                                                expansion_ratio=stage.get('expansion_ratio'),
                                                stochastic_depth_prob=current_stochastic_depth,
                                                SE_reduction_ratio=stage.get('SE_ratio'))
                        )
                        current_stochastic_depth += stochastic_depth_incremental
                    
                case 'AAP':
                    classifier_layers.append(nn.AdaptiveAvgPool2d(1))
                    
                case 'dropout':
                    prob = dropout_prob_override if (dropout_prob_override != -1) else stage.get('dropout_prob')
                    classifier_layers.append(nn.Flatten())
                    classifier_layers.append(nn.Dropout(p=prob, inplace=True))
                    
                case 'FC':
                    classifier_layers.append(nn.Linear(in_features=stage.get('in_features'),
                                        out_features=stage.get('out_features')))
            
            if len(layers) > 0:
                self.inner_stages.append(nn.Sequential(*layers))
                self.feature_vec = stage.get("out_channels")

                for layer in layers:
                    raw_param_names.extend([t[0] for t in layer.named_parameters()])

            layers: List[nn.Module] = []
        
        self.net = nn.Sequential(*self.inner_stages)
        self.classifier = nn.Sequential(*classifier_layers)

        for (name, _), raw_name in zip(self.named_parameters(),raw_param_names):
            self.layer_names.append(name.replace(raw_name,""))
        self.layer_names = list(unique_everseen(self.layer_names))

        self.apply(self._initialize_weights)


    def _initialize_weights(self, module: nn.Module):
        """Initialize weights for the CFGCNN module (to be used with .apply())"""
    
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Linear):
            init_range = 1.0 / math.sqrt(module.out_features)
            nn.init.uniform_(module.weight, -init_range, init_range)
            nn.init.zeros_(module.bias)


    def setDropoutProb(self, prob: float):
        """Setter for dropout probability"""
        self.classifier[2] = nn.Dropout(prob)


    def named_parameters_by_layer(self, layer: Union[str, int]) -> Iterator[Tuple[str, Parameter]]:
        """Return iterator over module's named parameters from a given layer 
        
        Args:
            layer_name: name (string) or index (int) of layer."""
        
        if isinstance(layer,int):
            layer = self.layer_names[layer]

        for name, param in self.named_parameters():
            if name.startswith(layer):
                yield (name,param)


    def forward(self, x):
       y = self.net(x)
       return self.classifier(y)
