"""
UdaciSense Project: Post-Training Quantization Module

This module provides utilities for applying post-training quantization to PyTorch models,
supporting both static and dynamic quantization methods.
"""

import os
import copy
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as models
import torch.ao.quantization as aoq
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx

from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
from torchvision.models.quantization.mobilenetv3 import _mobilenet_v3_conf, _mobilenet_v3_model


# TODO: Make MobileNetV3_Household model quantizable using stubs
# Consider whether you want to quantize the whole model or parts of it only
class QuantizableMobileNetV3_Household_ptq(nn.Module):
    """Quantizable MobileNetV3 model for household objects dataset.
    
    This model is designed to be compatible with PyTorch's quantization features,
    including quantization-aware training (QAT).
    
    Attributes:
        model: The underlying MobileNetV3 model with a modified classifier
    """
    
    def __init__(
        self, 
        num_classes: int = 10, 
        dropout_rate: float = 0.2, 
        quantize: bool = False, 
        pretrained: bool = True
    ):
        """Initialize a quantizable MobileNetV3 model.
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout probability in the classifier
            quantize: Whether to create a quantization-ready model
            pretrained: Whether to load ImageNet pretrained weights
        """
        super().__init__()
        # Create a quantizable MobileNetV3 Small
        inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small")
        self.model = _mobilenet_v3_model(
            inverted_residual_setting=inverted_residual_setting,
            last_channel=last_channel,
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None,
            progress=True,
            quantize=quantize,
        )
        
        # Modify the classifier for the household objects dataset
        last_channel = self.model.classifier[0].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(last_channel, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(1024, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Output tensor of shape [B, num_classes]
        """
        # Resize the image to the format expected by MobileNetV3
        x = torch.nn.functional.interpolate(
            x, size=(224, 224), mode='bilinear', align_corners=False
        )
        return self.model(x)
    
    # TODO: Fuse the model 
    def fuse_model(self) -> None:
        """
        Fuse conv, bn, relu layers for better quantization results
        Args:
        model: Model to fuse
        """
        print("Fusing layers...")

        # Get list of modules to fuse
        #modules_to_fuse = []
        # TODO: Identify patterns to fuse (Conv+BN, Conv+BN+ReLU, etc.)
        for m in self.modules():
            if m._get_name() == 'Conv2dNormActivation':
                modules_to_fuse = ["0", "1"]
                if len(m) == 3 and type(m[2]) is nn.ReLU:
                    modules_to_fuse.append("2")
                torch.ao.quantization.fuse_modules_qat(m, modules_to_fuse, inplace=True)
        
    

        

def quantize_model(
    model: nn.Module,
    calibration_data_loader: Optional[DataLoader] = None,
    calibration_num_batches: Optional[int] = None,
    quantization_type: str = "dynamic",
    backend: str = "fbgemm",
) -> nn.Module:
    """Apply post-training quantization to a PyTorch model.
    
    Args:
        model: The original model to quantize
        calibration_data_loader: DataLoader for calibration data,
            required for static quantization
        calibration_num_batches: Number of batches to run calibration on
        quantization_type: Type of quantization to apply:
            - "dynamic": Dynamic quantization (weights are quantized, activations quantized during inference)
            - "static": Static quantization (weights and activations are pre-quantized)
        backend: Quantization backend, either "fbgemm" (x86) or "qnnpack" (ARM)
            
    Returns:
        Quantized model
        
    Raises:
        ValueError: If an unsupported backend or quantization type is specified,
                   or if static quantization is requested without calibration data
    """
    # Verify backend
    if backend not in ["fbgemm", "qnnpack"]:
        raise ValueError("Backend must be either 'fbgemm' (x86) or 'qnnpack' (ARM)")
    
    # Create a copy of the model for quantization
    model_to_quantize = copy.deepcopy(model)
    
    # Set model to evaluation mode
    model_to_quantize.eval()
    
    # NOTE: Feel free to not implement all quantization types
    # Apply quantization based on type
    if quantization_type.lower() == "dynamic":
        return _apply_dynamic_quantization(model_to_quantize)
    elif quantization_type.lower() == "static":
        if calibration_data_loader is None:
            raise ValueError("Static quantization requires a calibration_data_loader")
        return _apply_static_quantization(model_to_quantize, calibration_data_loader, calibration_num_batches, backend)
    else:
        raise ValueError(f"Unsupported quantization type: {quantization_type}")

# TODO: Implement dynamic quantization, if selected
# Remember to look at built-in pytorch functionalities whenever possible
def _apply_dynamic_quantization(
    model: nn.Module
) -> nn.Module:
    """Apply dynamic quantization to a model.
    
    Dynamic quantization quantizes weights ahead of time but quantizes activations
    dynamically during inference.
    
    Args:
        model: Model to quantize (in eval mode)
        
    Returns:
        Dynamically quantized model
    """
    quantized_dynamic_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
    quantized_dynamic_model.eval()

    return quantized_dynamic_model
                

# TODO: Implement static quantization, if selected
# Remember to look at built-in pytorch functionalities whenever possible
# And that you first need to prepare the model for quantization, then apply calibration, and finally convert the model to quantized
def _apply_static_quantization(
    model: nn.Module,
    calibration_data_loader: DataLoader,
    calibration_num_batches: Optional[int] = None,
    backend: str = "fbgemm",
) -> nn.Module:
    """Apply static quantization to a model using provided calibration data.
    
    Static quantization quantizes both weights and activations ahead of time.
    
    Args:
        model: Model to quantize (in eval mode)
        calibration_data_loader: DataLoader for calibration data
        calibration_num_batches: Number of batches to use for calibration
        backend: Quantization backend, either "fbgemm" (x86) or "qnnpack" (ARM)
        
    Returns:
        Statically quantized model
    """
    print("Applying static quantization...")
    
    # If calibration_num_batches is not specified, use all available batches
    if calibration_num_batches is None:
        calibration_num_batches = len(calibration_data_loader)
    quantized_static_model = QuantizableMobileNetV3_Household_ptq()
    quantized_static_model = quantized_static_model.load_state_dict(torch.load("../models/baseline_mobilenet/checkpoints/model.pth", weights_only=True, map_location=torch.device('cpu')))
    print(quantized_static_model)
    quantized_static_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(quantized_static_model, inplace=True)
    print(quantized_static_model)

    def calibrate(model, data_loader):
        print("Starting Model calibration ....")
        model.eval()
        #steps = 0
        with torch.no_grad():
            for inp, lab in data_loader:
                model(inp)
        #        steps += 1
        #        if steps > 20:
        #            break
        print("Model calibration completed successfully ...")
        return model
        

    quantized_static_model = calibrate(quantized_static_model, calibration_data_loader)
    print(quantized_static_model)
    print("Running full static quantization now..")
    torch.quantization.convert(quantized_static_model, inplace=True)
    quantized_static_model.eval()
    print("Static Quantization completed successfully..")
    print(quantized_static_model)
    return quantized_static_model