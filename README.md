# Dataset and Analysis Scripts for Experimental Evaluation of LoRa Modulation using E22-900T22S over the Ocean Surface

This repository contains the dataset and analysis scripts from the experimental evaluation of LoRa modulation using the E22-900T22S module over the ocean surface, between a buoy-mounted node and a ground station.
The dataset includes both raw and processed data, along with the scripts used to generate the figures and results presented in the paper:

**F. D. Pacheco, A. F. Pinto, J. Maravalhas-Silva, B. M. Ferreira, N. A. Cruz** \
Experimental Evaluation of LoRa Communication over the Ocean Surface \
OCEANS 2025 - Great Lakes, Navy Pier, Chicago, United States, 2025 \
DOI: [to be updated] 

BibTeX citation:
```
@inproceedings{Pacheco2025EELoRaCommsOcean,
  author={F. D. Pacheco, A. F. Pinto, J. Maravalhas-Silva, B. M. Ferreira, N. A. Cruz},
  booktitle={{OCEANS 2025 - Great Lakes}}, 
  title={{Experimental Evaluation of LoRa Communication over the Ocean Surface}}, 
  year={2025},
  pages={},
  keywords={},
  doi={}
}
```

If you use this dataset or scripts, please cite the above paper.

## Dataset Description

**data_raw/**: Contains unprocessed logs directly captured from the experiments. 
- File naming: `mixip-node<1><2>-<3>-<4>-<5>-<6>.csv` 
  - 1: Node number 
  - 2: Measured transceiver role (tx/rx)
  - 3: Position number
  - 4: Transmission power level (10/17/22) [dBm]
  - 5: Wireless data rate (2.4/4.8/9.6/19.2) [kbps]
  - 6: LoRa payload packet size (32) [B]
- File CSV structure (left-to-right, no headers):
  - Index: row index
  - Timestamp: sample measured time [hh:mm:ss.uuuuuu]
  - RSSI: measured received signal strength [dBm] (*)
  - Noise: measured noise signal strength [dBm] (*)
  - RP index: received packet index assigned by the RX
  - TP index: transmitted packet index assigned by the TX
  - Lat: GPS latitude of the measurment [DD]
  - Lon: GPS longitude of the measurment [DD]

* values require conversion using: -(256 + value x 2) due to a measurement-time conversion bug.
    
**data_processed/**: Processed raw data 
- File naming: `<1>-<2>-<3>.csv`
  - 1: Wireless data rate (2.4/4.8/9.6/19.2) [kbps] 
  - 2: Transmission power level (10/17/22) [dBm]
  - 3: LoRa payload packet size (32) [B]
    
- File CSV structure (left-to-right), has headers:
  - `subindex`: row index
  - `time`: timestamp [YYYY-MM-DD hh:mm:ss.uuuuuu]
  - `rssi`: measured RSSI correctly converted [dBm]
  - `noise`: measured noise floor level correctly converted [dBm]
  - `nrx`: index of the received packet, assigned by RX
  - `ntx`: index of the transmitted packet, assigned by TX
  - `lat`: GPS latitude of the measurment [DD]
  - `lon`: GPS longitude of the measurment [DD]
  - `node`: Node number
  - `role`: Measured transceiver role (tx/rx)
  - `pos`: Position number
  - `power`: Transmission power level (10/17/22) [dBm]
  - `rate`: Wireless data rate (2.4/4.8/9.6/19.2) [kbps] 
  - `size`: LoRa payload packet size (32) [B]
  - `distances`: geodesic distance between position 1 and current position (computed with geopy)

## Experimental Setup

Hardware of both nodes:
  - E22-900T22S LoRa module
  - 5 dBi omnidirectional antenna

Environment:
  - Over the ocean surface, one node mounted on a buoy, the other on a car roof parked by the coast with direct line-of-sight

Configuration Parameters:
  - Wireless data rate: 2.4, 4.8, 9.6, 19.2 kbps
  - Transmission power level: 10, 17, 22 dBm
  - Tested distances between nodes: 1, 297, 575, 1048, 1222, 1707, 2276, 2836 m

## Usage Instructions

Requirements:
  - Python ≥ 3.11.2
  - Dependencies: pandas, matplotlib, numpy, scipy, geopy

Running:
```
# Process raw data
python3 scripts/processRawData.py -p data_raw/ -d data_processed/ -t 1 -r 0

# Generate processed figures
cd figures
python3 ../scripts/generatePlotsAnalyzedData.py -p ../data_processed/

# Generate noise floor histogram
cd figures
python3 ../scripts/generateHistogramNoise.py -p ../data_processed/noise.csv
```

## License
- Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en)
- Code: [MIT License](https://opensource.org/license/mit)

## Contributing
If you find issues or have suggestions, please open an issue or submit a pull request.

## Acknowledgments
This work was partly funded by the SeaGuard project, which received funding from the European Union’s Horizon Europe Programme under Grant Agreement No. 101168489.
