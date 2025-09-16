# File:    generatePlotsAnalyzedData.py
# Author:  Fábio D. Pacheco
# Date:    16-09-2025

import argparse
import os
import re
import pandas
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def findFiles( path ):
  files = []

  if not os.path.isdir( path ):
    print(f"Path {path} is not a directory ...")
    return None
  
  regexPattern = r".*\.csv"
  cregex = re.compile( regexPattern )

  for filename in os.listdir( path ):
    fullPath = os.path.join( path, filename )
 
    if os.path.isfile( fullPath ):
      if cregex.search( filename ):
        print(f"Found {filename} ...")
        files.append( fullPath )
      
  return files

def parseFileName( path ):
  if os.path.isfile( path ):
    pattern = re.compile(r"^(\d+)-(\d+)-(\d+)\.csv$")
    match = pattern.match( os.path.basename(path) )

    if match:
      rate, power, packet = match.groups( )      
      return {
        'power': power,
        'rate': rate,
        'size': packet,
      }
  return None

def readFile( path ):
  if not os.path.isfile( path ):
    return None

  try:
    doc = pandas.read_csv( path )
    doc['time'] = pandas.to_datetime(doc['time'], errors='coerce')
    return doc

  except Exception as e:
    print(f"Error reading file {path}: {e}")
    return None

def onclick(event):
    # This function is called every time a mouse button is clicked
    if event.xdata is not None and event.ydata is not None:
        print(f"Clicked coordinates: x={event.xdata:.2f}, y={event.ydata:.2f}")


if __name__ == "__main__":
  arguments = argparse.ArgumentParser( )
  arguments.add_argument("-p", "--input", type=str, required=True, help="Path to input directory")
  args = arguments.parse_args( )

  pathFiles = findFiles( args.input )
  results = {}

  for path in pathFiles:
    name = f"r{ parseFileName(path)['rate'] }p{ parseFileName(path)['power'] }s{ parseFileName(path)['size'] }"
    print(name)
    results[ name ] = readFile( path )

  plt.style.use('seaborn-v0_8-whitegrid')
  plt.rcParams.update({
      "text.usetex": True,
      "font.family": "serif",
      "font.serif": ["Times New Roman"],
      "figure.figsize": (14, 5)
  })

  positions = [0, 1, 2, 3, 4, 5, 6, 7]
  air_rate = [2400, 4800, 9600, 19200]
  tx_power = [10, 17, 22]
  packet_size = 32

  fontSize = 19
  fontSizeUnits = 17
  sizeX = 6.5
  sizeY = 5.25
  lineSize = 2
  colors = plt.cm.get_cmap('tab10', len(air_rate))  
  colors_power = plt.cm.get_cmap('tab10', len(tx_power))  
  markers = ['o', 's', '^', 'p', 'h', 'D', 'v', '<', '>', 'X', 'P', '*']

  successRate = []
  for k, _position in enumerate(positions):
    _df = results['r2400p22s32']
    _dist_df = _df[(_df['pos'] == _position)]

    if not _dist_df.empty:
      _dist = _dist_df['distance'].iloc[0]

      for j, _power in enumerate(tx_power):
        for i, _rate in enumerate(air_rate):
          tx_name = f"r{_rate}p{_power}s{packet_size}"      

          if tx_name in results:
            result_df = results[tx_name]  

            result_rx_df = result_df[(result_df['role'] == 'rx') & (result_df['rssi'] > -128) & (result_df['pos'] == _position)]
            result_tx_df = result_df[(result_df['role'] == 'tx') & (result_df['pos'] == _position)]
        
            sr = 0
            dist = None
            try:
              dist = result_rx_df['distance'].iloc[0]
            except Exception as e:
              print(f"Error: {e}")

            if dist is not None:
              try:
                sr = round( result_rx_df['nrx'].iloc[-1] / result_tx_df['ntx'].iloc[-1] , 2 ) * 100

                successRate.append({
                  'success': sr, 
                  'distance': result_rx_df['distance'].iloc[0],
                  'rate': _rate,
                  'power': _power,
                  'rssi': result_rx_df['rssi'].mean( )
                })
              except Exception as e:
                print("Zero division")
            else:
              print(f"Distance: {_dist}")
              successRate.append({
                'success': sr, 
                'distance': _dist,
                'rate': _rate,
                'power': _power,
                'rssi': -100.7
              })           

              print(f"Success Rate ({_rate},{_power},{ result_df['distance'].iloc[0] }): {sr:.2f}%")

  successRate_df = pandas.DataFrame(successRate)

  for j, _power in enumerate(tx_power):
    fig = plt.figure(figsize=(sizeX, sizeY), dpi=300)
    for i, _rate in enumerate(air_rate):
      sr_df = successRate_df[(successRate_df['rate'] == _rate) & (successRate_df['power'] == _power)]

      sr_df = sr_df.sort_values('distance')
      sr_df['distance'] = sr_df['distance']

      plt.scatter(
        sr_df['distance'],
        sr_df['success'],
        color=colors(i),
        marker=markers[i % len(markers)],
        s=16,
        alpha=0.7
      )

      sr2_df = sr_df[ successRate_df['distance'] > 300 ]
      anchor = pandas.DataFrame([{'distance': 1, 'success': 100}])
      sr2_df = pandas.concat([anchor, sr2_df], ignore_index=True)

      plt.plot(
        sr2_df['distance'],
        sr2_df['success'],
        label=f'$R_b$ = {_rate/1000} [kbs$^{{-1}}$]',
        color=colors(i),
        linestyle='--', 
        linewidth=lineSize,
        marker=markers[(i) % len(markers)],
        markersize=4,
        alpha=0.7
      )

      filename = f"{_power}-{_rate}.csv"
      sr2_df.to_csv(filename, index=False)

    plt.xlabel('Distance [m]', fontsize=fontSize)
    plt.ylabel('Success rate [\%]', fontsize=fontSize)

    plt.axis((0, 3500, 0, 110))
    plt.legend(loc='best', fontsize=fontSizeUnits, facecolor='white', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='both', which='major', labelsize=fontSizeUnits, pad=10)
    plt.tight_layout()
    fig.canvas.mpl_connect('button_press_event', onclick)
    # plt.show( )

    output_filename = f'success_vs_distance_at{_power}.pdf'
    plt.savefig(output_filename, bbox_inches='tight')

    plt.close( )

  for j, _power in enumerate(tx_power):
    fig = plt.figure(figsize=(sizeX, sizeY), dpi=300)
    for i, _rate in enumerate(air_rate):
      tx_name = f"r{_rate}p{_power}s{packet_size}"
      
      if tx_name in results:
        result_df = results[tx_name]
        result_df['rssi'] = result_df['rssi'].astype(float)
        df = result_df[(result_df['role'] == 'rx') & (result_df['rssi'] > -128)]

        grouped_df = df.groupby('distance').agg(
          mrssi = ('rssi', 'mean'),
          srssi = ('rssi', 'std')
        ).reset_index( )

        plt.errorbar(
          x=grouped_df['distance'],
          y=grouped_df['mrssi'],
          yerr=grouped_df['srssi'],  
          label=f'$R_b$ = {_rate/1000} [kbs$^{{-1}}$]',
          fmt=markers[i % len(markers)], 
          color=colors(i),
          capsize=5, 
          elinewidth=lineSize, 
          # s=4,
          alpha=0.7
        )

        grouped_df['distance'] = grouped_df['distance'].astype(float)
        grouped_df = grouped_df[ grouped_df['distance'] > 400 ]
        
        anchor_point = pandas.DataFrame([{'distance': 1, 'mrssi': 0, 'srssi': 0}])
        grouped_df = pandas.concat([anchor_point, grouped_df], ignore_index=True)

        x_trend = np.linspace(0.1, max(grouped_df['distance']), 100)

        def log_distance_model(d, a, n):
          """Log-Distance Path Loss Model"""
          return a - 10 * n * np.log10(d)

        x_data = grouped_df['distance']
        y_data = grouped_df['mrssi']

        popt, pcov = curve_fit(log_distance_model, x_data, y_data)
        a, n = popt
        print(f"The fitted Log-Distance Path Loss model is: (for, power={_power}, rate={_rate}) L(d) = {a:.2f} - 10 * {n:.2f} * log10(d)")

        # Calculate R-squared
        y_predicted = log_distance_model(x_data, a, n)
        ss_res = np.sum((y_data - y_predicted) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"R-squared value: {r_squared:.4f}")

        plt.plot(x_trend, log_distance_model(x_trend, a, n), linestyle='--', color=colors(i), linewidth=lineSize)

      else:
        print(f"Skipping plot for '{tx_name}', data not found.")


    # plt.title(f'RSSI vs. Distance for {_power} [dBm]', fontsize=16)
    plt.xlabel('Distance [m]', fontsize=fontSize)
    plt.ylabel('RSSI [dBm]', fontsize=fontSize)
    
    plt.legend(loc='best', fontsize=fontSizeUnits, facecolor='white', frameon=True)
    plt.axis((0, 3000, -120, -10))
    plt.tick_params(axis='both', which='major', labelsize=fontSizeUnits, pad=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    fig.canvas.mpl_connect('button_press_event', onclick)
    # plt.show( )

    output_filename = f'rssi_vs_distance_at{_power}.pdf'
    plt.savefig(output_filename, bbox_inches='tight')
    
    plt.close( )
      
  models = ["FSPL", "TRPL"]
  for model in models:
    fig = plt.figure(figsize=(sizeX, sizeY), dpi=300)
    for i, _power in enumerate(tx_power):
      Gt_dBi = 5
      Gr_dBi = 5
      
      def fspl_model(d, f_hz, tx_power_dbm, Gt_dBi, Gr_dBi, c=3e8):
        """
        FSPL model to calculate received power (Prx) in dBm.
        """
        fspl_db = 20 * np.log10(d) + 20 * np.log10(f_hz) - 20 * np.log10(c) + 20 * np.log10(4 * np.pi)          
        prx_dbm = tx_power_dbm + Gt_dBi + Gr_dBi - fspl_db
        return prx_dbm

      f_hz = 868 * 1e6

      def two_ray_pl_model(d, ht, hr, tx_power_dbm, Gt_dBi, Gr_dBi):
        """
        Two-Ray Path Loss Model to calculate received power (Prx) in dBm.
        Assumes Gt and Gr are linear gains.
        """
        Gt = 10**(Gt_dBi / 10)
        Gr = 10**(Gr_dBi / 10)

        return np.where(
          d == 0,
          tx_power_dbm + Gt_dBi + Gr_dBi,
          10 * np.log10(Gt * Gr * (ht * hr)**2 / d**4 * 1000) + tx_power_dbm 
        )

      ht = 1  # m
      hr = 3  # m

      # FSPL Model Plot
      if model == "FSPL":
        plt.plot(x_trend, fspl_model(x_trend, f_hz, _power, Gt_dBi, Gr_dBi), 
                label=f'$P_t$ = {_power} [dBm]', linestyle='--', linewidth=lineSize, color=colors_power(i))

      else:
        # Two-Ray PL Model Plot
        plt.plot(x_trend, two_ray_pl_model(x_trend, ht, hr, _power, Gt_dBi, Gr_dBi), 
                label=f'$P_t$ = {_power} [dBm]', linestyle='--', linewidth=lineSize, color=colors_power(i))

    plt.xlabel('Distance [m]', fontsize=fontSize)
    plt.ylabel('RSSI [dBm]', fontsize=fontSize)
    
    plt.legend(loc='best', fontsize=fontSizeUnits, facecolor='white', frameon=True)
    plt.axis((0, 3000, -120, -10))
    plt.tick_params(axis='both', which='major', labelsize=fontSizeUnits, pad=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    fig.canvas.mpl_connect('button_press_event', onclick)
    # plt.show( )

    output_filename = f'model{model}_rssi_vs_distance_at{_power}.pdf'
    plt.savefig(output_filename, bbox_inches='tight')
    
    plt.close( )

  throughput = []
  for k, _position in enumerate(positions):
    _df = results['r2400p22s32']
    _dist_df = _df[(_df['pos'] == _position)]

    if not _dist_df.empty:
      _dist = _dist_df['distance'].iloc[0]

      for j, _power in enumerate(tx_power):
        fig = plt.figure(figsize=(sizeX, sizeY), dpi=300)
        for i, _rate in enumerate(air_rate):
          tx_name = f"r{_rate}p{_power}s{packet_size}"      

          if tx_name in results:
            result_df = results[tx_name]
            df = result_df[(result_df['role'] == 'rx') & (result_df['pos'] == _position)]
          
            df = df.sort_values('nrx')
            bk = df['nrx'].reset_index(drop=True)
            df['eltime'] = (df['time'] - df['time'].min()).dt.total_seconds()
            df = df.sort_values('eltime').reset_index(drop=True)
            df['nrx'] = bk
            df['nrx'] = df['nrx'].astype(float)
            df['nrx'] = df['nrx'] * 8 * packet_size

            plt.scatter(
              df['eltime'],
              df['nrx'] / 1000,
              label=f'$R_b$ = {_rate / 1000} [kbs$^{{-1}}$]',
              color=colors(i),
              marker=markers[i % len(markers)],
              s=16,
              alpha=0.7
            )

            x_data = df['eltime']
            y_data = df['nrx']
            if not x_data.empty and not y_data.empty:
              print(_rate)
              coefficients = np.polyfit(x_data, y_data, deg=1)
              poly_function = np.poly1d(coefficients)

              throughput.append({
                'distance': df['distance'].iloc[0],
                'throughput': coefficients[0],
                'rate': _rate,
                'power': _power
              })

              print(f"Coefficient({_rate},{_power}) = {coefficients[0]}")
            else:
              throughput.append({
                'distance': _dist,
                'throughput': 0,
                'rate': _rate,
                'power': _power
              })

          else:
            print(f"Skipping plot for '{tx_name}', data not found.")

        plt.xlabel('Time [s]', fontsize=fontSize)
        plt.ylabel('$\sum$ Throughput [kb]', fontsize=fontSize)
        
        plt.legend(loc='best', fontsize=fontSizeUnits, facecolor='white', frameon=True, markerscale=2.0)
        # plt.axis((0, 35, 0, 350))
        plt.tick_params(axis='both', which='major', labelsize=fontSizeUnits, pad=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        fig.canvas.mpl_connect('button_press_event', onclick)
        # plt.show( )

        output_filename = f'cum_throughput_vs_time_at{_power,_position}.pdf'
        plt.savefig(output_filename, bbox_inches='tight')

      plt.close( )

  
  throughput_df = pandas.DataFrame(throughput)

  for j, _power in enumerate(tx_power):
    fig = plt.figure(figsize=(sizeX, sizeY), dpi=300)
    for i, _rate in enumerate(air_rate):
      rate_df = throughput_df[(throughput_df['rate'] == _rate) & (throughput_df['power'] == _power)]

      rate_df = rate_df.sort_values('distance')
      rate_df['throughput'] = rate_df['throughput'].astype(float)
      rate_df['throughput'] = rate_df['throughput'] / 1000

      plt.scatter(
        rate_df['distance'],
        rate_df['throughput'],
        # label=f'$R_b$ = {_rate} [bs$^{{-1}}$]',
        color=colors(i),
        marker=markers[i % len(markers)],
        s=16,
        alpha=0.7
      )

      _rate_df = rate_df[ rate_df['distance'] > 300 ]
      if not _rate_df.empty:
        x_data = _rate_df['distance']
        y_data = _rate_df['throughput']
        coefficients = np.polyfit(x_data, y_data, deg=2)
        poly_function = np.poly1d(coefficients)
        x_trend = np.linspace(min(_rate_df['distance']), max(_rate_df['distance']), 100)
        # plt.plot(x_trend, poly_function(x_trend), linestyle='--', color=colors(i), linewidth=1)

        plt.plot(
          _rate_df['distance'],
          _rate_df['throughput'],
          label=f'$R_b$ = {_rate / 1000} [kbs$^{{-1}}$]',
          color=colors(i),
          linestyle='--', 
          linewidth=lineSize,
          marker=markers[(i) % len(markers)],
          markersize=4,
          alpha=0.7
        )

    plt.xlabel('Distance [m]', fontsize=fontSize)
    plt.ylabel('Throughput [kbs$^{-1}$]', fontsize=fontSize)

    plt.axis((0, 3500, 0, 6))
    plt.legend(loc='best', fontsize=fontSizeUnits, facecolor='white', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='both', which='major', labelsize=fontSizeUnits, pad=10)
    plt.tight_layout()
    fig.canvas.mpl_connect('button_press_event', onclick)
    # plt.show( )

    output_filename = f'throughput_vs_distance_at{_power}.pdf'
    plt.savefig(output_filename, bbox_inches='tight')

    plt.close( )


  for j, _power in enumerate(tx_power):
    fig = plt.figure(figsize=(sizeX, sizeY), dpi=300)
    for i, _rate in enumerate(air_rate):
      sr_df = successRate_df[(successRate_df['rate'] == _rate) & (successRate_df['power'] == _power)]

      noise = -100.7
      sr_df['snr'] = sr_df['rssi'] - noise

      sr_df = sr_df.sort_values('snr')
      plt.scatter(
        sr_df['snr'],
        100 - sr_df['success'],
        color=colors(i),
        marker=markers[i % len(markers)],
        s=16,
        alpha=0.7
      )

      plt.plot(
        sr_df['snr'],
        100 - sr_df['success'],
        label=f'$R_b$ = {_rate/1000} [kbs$^{{-1}}$]',
        color=colors(i),
        linestyle='--', 
        linewidth=lineSize,
        marker=markers[(i) % len(markers)],
        markersize=4,
        alpha=0.7
      )

    plt.xlabel('SNR [dB]', fontsize=fontSize)
    plt.ylabel('Loss rate [\%]', fontsize=fontSize)
    plt.legend(loc='best', fontsize=fontSizeUnits, facecolor='white', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='both', which='major', labelsize=fontSizeUnits, pad=10)
    plt.tight_layout()
    fig.canvas.mpl_connect('button_press_event', onclick)
    # plt.show( )

    output_filename = f'success_vs_rssi_at{_power}.pdf'
    plt.savefig(output_filename, bbox_inches='tight')

    plt.close( )