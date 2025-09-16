# File:    generateHistogramNoise.py
# Author:  Fábio D. Pacheco
# Date:    16-09-2025

import pandas as pd
import matplotlib.pyplot as plt
import argparse

if __name__ == "__main__":
  arguments = argparse.ArgumentParser( )
  arguments.add_argument("-p", "--input", type=str, required=True, help="Path to input file")
  args = arguments.parse_args( )

  try:
    df = pd.read_csv( args.input, header=None, names=['timestamp', 'rssi', 'lat', 'lon'])
  
    df.loc[df['rssi'] > 0, 'rssi'] = df.loc[df['rssi'] > 0, 'rssi'] * -1
    df['rssi'] = -(256 + 2 * df['rssi']) 
    print(f"Mean:{df['rssi'].mean()}")
    print(f"Standard Deviation:{df['rssi'].std()}")

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "figure.figsize": (14, 5)
    })

    positions   = [0, 1, 2, 3, 4, 5, 6, 7]
    air_rate    = [2400, 4800, 9600, 19200]
    tx_power    = [10, 17, 22]
    packet_size = 32

    fontSize = 19
    fontSizeUnits = 17
    sizeX = 6.5
    sizeY = 5.25
    lineSize = 2
    colors = plt.cm.get_cmap('seismic', len(air_rate))  
    markers = ['o', 's', '^', 'p', 'h', 'D', 'v', '<', '>', 'X', 'P', '*']

    plt.figure(figsize=(sizeX, sizeY), dpi=300)

    plt.hist(df['rssi'], bins=6, edgecolor='black', alpha=0.7, color='grey' )
    plt.xlabel('RSSI [dBm]', fontsize=fontSize)
    plt.ylabel('Frequency', fontsize=fontSize)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='both', which='major', labelsize=fontSizeUnits)
    plt.tight_layout()

    plt.savefig('rssi_histogram.pdf')
    print("Histogram saved as rssi_histogram.pdf")
    plt.show( )

    output_filename = f'noise_histogram.pdf'
    plt.savefig(output_filename, bbox_inches='tight')

    plt.close( )

  except Exception as e:
    print(f"Failed: {e}")