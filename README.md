# ConvLSTM+InceptionS1S2

ConvLSTM+InceptionS1S2 is a deep learning model designed for multimodal satellite image segmentation. This model leverages spatiotemporal data from Sentinel-2 (S2) and Sentinel-1 (S1) satellite images to perform segmentation tasks. It combines ConvLSTM layers, an Inception module, and a U-Net architecture for feature extraction and segmentation.

## Features
- **ConvLSTM for Spatiotemporal Data**: Processes sequential data from Sentinel-2 and Sentinel-1 to capture temporal and spatial dependencies.
- **Inception Module**: Extracts multi-scale features from contextual stack input.
- **U-Net Architecture**: Performs final segmentation with high accuracy.
- **Multimodal Data Integration**: Combines data from multiple sources (e.g., S1, S2, and stack input) for robust analysis.

## Architecture
The model consists of three main components:
1. **ConvLSTM Layers**:
   - Separate ConvLSTM layers for Sentinel-2 and Sentinel-1 data.
   - Captures temporal relationships in the sequence of satellite images.
2. **Inception Module**:
   - Extracts multi-scale features from additional contextual data.
3. **U-Net**:
   - Combines features from ConvLSTM and Inception for segmentation.
   - Uses a frozen VGG16 encoder to reduce computational load.

## Example Dataset
This model is designed for satellite image segmentation. Ensure that your dataset includes:
- **Sentinel-2 (S2)**: Multispectral image sequences.
- **Sentinel-1 (S1)**: SAR image sequences.
- **Contextual Data**: First S1 and S2 date (please refer to the paper for this part)

Data preprocessing should ensure consistent spatial resolution and temporal alignment of inputs. This model was created for MultiSenGE dataset. You can also use [MultiSenNA](https://doi.theia.data-terra.org/ai4lcc/?lang=en) dataset.

## Model Requirements
- **Input Dimensions**:
  - `x_s2`: (batch_size, sequence_length, height, width, channels=10).
  - `x_s1`: (batch_size, sequence_length, height, width, channels=2).
  - `x_stack`: (batch_size, height, width, channels=12).
- **Output Dimensions**:
  - Segmentation map: (batch_size, height, width, classes).

## Dependencies
- PyTorch
- segmentation_models_pytorch
- NumPy
- Matplotlib
- GDAL (if needed)

## Citation
If you use this model in your research, please cite:
```
@article{wenger2022multimodal,
  title={Multimodal and multitemporal land use/land cover semantic segmentation on sentinel-1 and sentinel-2 imagery: An application on a multisenge dataset},
  author={Wenger, Romain and Puissant, Anne and Weber, Jonathan and Idoumghar, Lhassane and Forestier, Germain},
  journal={Remote Sensing},
  volume={15},
  number={1},
  pages={151},
  year={2022},
  publisher={MDPI}
}

```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
