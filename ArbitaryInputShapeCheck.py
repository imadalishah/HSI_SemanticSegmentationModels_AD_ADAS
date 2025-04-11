import torch
import matplotlib.pyplot as plt
import numpy as np
from models import *

# Test with the arbaterary input dimensions
def test_models():
    # Define the test input dimensions
    test_inputs = [
        (15, 210, 150),   # [C, H, W]
        (128, 472, 355),  # [C, H, W]
        (25, 209, 416)    # [C, H, W]
    ]
    
    # Define the models to test
    model_types = [
        'UNet', 
        'UNet_CBAM', 
        'UNet_CA', 
        'UNet_CBAM_SkipConnection',
        'DeepLabV3Plus',
        'HRNet',
        'PSPNet'
    ]
    
    # Set output channels (classes) for all models
    out_channels = 10
    
    # Create a figure for visualization
    fig, axes = plt.subplots(len(test_inputs), len(model_types), figsize=(20, 12))
    
    # Test each model with each input dimension
    for i, (in_channels, height, width) in enumerate(test_inputs):
        print(f"\nTesting input with dimensions: [C={in_channels}, H={height}, W={width}]")
        
        # Create a random input tensor
        x = torch.randn(1, in_channels, height, width)
        
        for j, model_type in enumerate(model_types):
            print(f"  Testing {model_type}...")
            
            try:
                # Create the model
                model = create_model(model_type, in_channels, out_channels)
                
                # Set model to evaluation mode
                model.eval()
                
                # Forward pass
                with torch.no_grad():
                    output = model(x)
                
                # Print output shape
                print(f"    Success! Output shape: {output.shape}")
                
                # Visualize a slice of the output (first channel)
                if isinstance(output, tuple):
                    output = output[0]
                
                # Plot the output
                ax = axes[i, j]
                im = ax.imshow(output[0, 0].numpy(), cmap='viridis')
                ax.set_title(f"{model_type}\n{output.shape}")
                ax.axis('off')
                
            except Exception as e:
                print(f"    Failed: {str(e)}")
                
                # Show error in the plot
                ax = axes[i, j]
                ax.text(0.5, 0.5, f"Error: {str(e)}", 
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax.transAxes,
                        wrap=True)
                ax.set_title(f"{model_type} - Failed")
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('model_test_results.png')
    plt.close()
    
    return "All tests completed. Check visualization for verification."

# Run the tests
result = test_models()
print(result)

# Display the saved visualization
img = plt.imread('model_test_results.png')
plt.figure(figsize=(20, 12))
plt.imshow(img)
plt.axis('off')
plt.show()
