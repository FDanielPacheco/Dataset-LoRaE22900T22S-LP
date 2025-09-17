# File:    processRawData.py
# Author:  Fábio D. Pacheco
# Date:    16-09-2025

import argparse
import os
import re
import pandas
from geopy.distance import geodesic

# Save headers
CSV_HEADERS_1 = [
  'Index', 'SubIndex', 'Node', 'Role', 'Timestamp [HH:MM:SS.UUUUUU]', 'Position', 'Latitude [º]', 'Longitude [º]', 'Distance between the 2 nodes [m]', 'Transmission Power [dBm]', 'Air Data Rate [bps]', 'Packet Size [B]', 'RSSI [dBm]', '# TX Segments', '# RX Segments'
]

PD_HEADERS_1 = [
  'index', 'subindex', 'node', 'role', 'time', 'pos', 'lat', 'lon', 'distance', 'power', 'rate', 'size', 'rssi', 'ntx', 'nrx'
]

CSV_HEADERS_2 = [
  'Index', 'SubIndex', 'Node', 'Timestamp [HH:MM:SS.UUUUUU]', 'Latitude [º]', 'Longitude [º]', 'Noise [dBm]'
]

PD_HEADERS_2 = [
  'index', 'subindex', 'node', 'time', 'lat', 'lon', 'noise'
]

# Aliases
INDEX       = PD_HEADERS_1[0]
SUBINDEX    = PD_HEADERS_1[1]
NODE        = PD_HEADERS_1[2]
ROLE        = PD_HEADERS_1[3]
TIME        = PD_HEADERS_1[4]
POSITION    = PD_HEADERS_1[5]
LAT         = PD_HEADERS_1[6]
LON         = PD_HEADERS_1[7]
DISTANCE    = PD_HEADERS_1[8]
POWER       = PD_HEADERS_1[9]
RATE        = PD_HEADERS_1[10]
SIZE        = PD_HEADERS_1[11]
RSSI        = PD_HEADERS_1[12]
TX_SEGMENTS = PD_HEADERS_1[13]
RX_SEGMENTS = PD_HEADERS_1[14]
NOISE       = PD_HEADERS_2[6]

def findFiles( path, key, role ):
  files = []

  if not os.path.isdir( path ):
    print(f"Path {path} is not a directory ...")
    return None
  
  regexPattern = rf"mixip-node{key}{role}-.*\.csv"
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
    pattern = re.compile(r"^mixip-node(\d+)(tx|rx)-(\d+)-([+-]?\d+)-(\d+)-(\d+)(?:-(.*))?\.csv$")
    match = pattern.match( os.path.basename(path) )

    if match:
      node, role, pos, power, rate, packet, extra = match.groups( )
      extraField = extra if extra is not None else ""
      
      return {
        'type': "noise" if "noise" == extraField else "comms",
        'role': role,
        'node': node,
        'pos': pos,
        'power': power,
        'rate': rate,
        'size': packet,
      }
  return None

def readFile( path ):
  if not os.path.isfile( path ):
    return None

  info = parseFileName( path )
  if info is None:
    return None

  try:
    doc = pandas.read_csv( path, header=None )
  
    # Original headers
    headers = {
      0: SUBINDEX, 
      1: TIME, 
      2: RSSI, 
      3: NOISE, 
      4: RX_SEGMENTS,
      5: TX_SEGMENTS, 
      6: LAT, 
      7: LON,
    }
    
    doc = doc.rename( columns=headers )
    info = parseFileName( path )

    if info['type'] == "comms":
      doc.rename(columns={
        0: SUBINDEX, 
        1: TIME, 
        2: RSSI, 
        3: NOISE, 
        4: RX_SEGMENTS, 
        5: TX_SEGMENTS,
        6: LAT, 
        7: LON,
      }, inplace=True)

      doc[TIME] = pandas.to_datetime(doc[TIME], format='%H:%M:%S.%f')
          
      doc[RSSI] = doc[RSSI].astype(float)

      zeroMask = doc[RSSI] == 0
      doc[RX_SEGMENTS] = doc[RX_SEGMENTS].fillna(method='ffill')
      doc.loc[zeroMask, RX_SEGMENTS] = pandas.NA
      doc = doc[~zeroMask].copy()
      doc = doc.reset_index(drop=True)
      doc[RSSI] = doc[RSSI].astype(float)
      doc[RSSI] = -(256 + doc[RSSI] * 2) 
      
    elif info['type'] == "noise":
      doc.rename(columns={
        0: SUBINDEX, 
        1: TIME, 
        2: LAT, 
        3: LON, 
        4: NOISE,
      }, inplace=True)
      
      doc[TIME] = pandas.to_datetime(doc[TIME], format='%H:%M:%S.%f')
      doc[NOISE] = doc[NOISE].astype(float)

    nrows = len(doc)
    doc.loc[:, NODE]     = [info['node']]  * nrows
    doc.loc[:, ROLE]     = [info['role']]  * nrows
    doc.loc[:, POSITION] = [info['pos']]   * nrows
    doc.loc[:, POWER]    = [info['power']] * nrows
    doc.loc[:, RATE]     = [info['rate']]  * nrows
    doc.loc[:, SIZE]     = [info['size']]  * nrows  

    return doc

  except Exception as e:
    print(f"Error reading file {path}: {e}")
    return None



def correctCoordinates( data ):
  correctCoordinates = {
    0: (40.788899, -8.671858),
    1: (40.784590, -8.675635),
    2: (40.780168, -8.680668), 
    3: (40.790465, -8.674706),
    4: (40.785130, -8.683242),
    5: (40.780470, -8.688765),
    6: (40.775590, -8.692361),
    7: (40.770278, -8.694880),
  }  

  def applyCorrection( group ):
    if group.empty:
      return group

    position_key = int(group[POSITION].iloc[0])

    rxNodeMask = group[ROLE] == "rx"
    txNodeMask = group[ROLE] == "tx"

    if rxNodeMask.any( ):
      lat, lon = correctCoordinates[0]
      group.loc[rxNodeMask, LAT] = lat
      group.loc[rxNodeMask, LON] = lon

    if txNodeMask.any( ) and (position_key + 1) in correctCoordinates:
      lat, lon = correctCoordinates[position_key + 1]
      group.loc[txNodeMask, LAT] = lat
      group.loc[txNodeMask, LON] = lon
    return group

  return data.groupby(POSITION).apply(applyCorrection).reset_index(drop=True)

def calculateDistance( data ):
  def calculateForGroup( group ):
    rxNodeMask = group[ROLE] == "rx" 
    rxNode = group[ rxNodeMask ]
    txNodeMask = group[ROLE] == "tx"
    
    if not rxNode.empty and txNodeMask.any():
      rxCoords = (rxNode[LAT].iloc[0], rxNode[LON].iloc[0])
      txCoords = (group[txNodeMask][LAT].iloc[0], group[txNodeMask][LON].iloc[0])

      distance = geodesic( rxCoords, txCoords ).m 
      group.loc[txNodeMask, DISTANCE] = distance
      group.loc[rxNodeMask, DISTANCE] = distance

    return group
  return data.groupby(POSITION).apply(calculateForGroup).reset_index(drop=True)

def separateDataFrames( data ):
  params = data.groupby([RATE, POWER, SIZE]).size( ).reset_index( )

  separetedDataName = {}
  for _, row in params.iterrows( ):
    rate = row[RATE]
    power = row[POWER]
    size = row[SIZE]
    
    dataFiltered = data[
      (data[RATE] == rate)   & 
      (data[POWER] == power) & 
      (data[SIZE] == size)
    ].copy().reset_index(drop=True)
    
    dataName = f"{rate}-{power}-{size}"
    separetedDataName[dataName] = dataFiltered
    print(f" - Created DataFrame: {dataName} (with {len(dataFiltered)} rows)")

  return separetedDataName

def timeShift( separatedData ):
  updateData = {}
  flag = False

  for name, data in separatedData.items( ):
    data[POSITION] = data[POSITION].astype(float)
    rxPos0 = data[( data[POSITION] == 0 ) & (data[ROLE] == "rx")].copy( )
    txPos0 = data[( data[POSITION] == 0 ) & (data[ROLE] == "tx")].copy( )

    shiftLatency = pandas.NaT
    shiftClocks = pandas.NaT    

    if not rxPos0.empty and not txPos0.empty :
      rx_s0_filtered = rxPos0[rxPos0[SUBINDEX] == 0]
      tx_s0_filtered = txPos0[txPos0[SUBINDEX] == 0]
      rx_s1_filtered = rxPos0[rxPos0[SUBINDEX] == 1]

      if not rx_s0_filtered.empty and not tx_s0_filtered.empty and not rx_s1_filtered.empty:
        rx_time_s0 = rx_s0_filtered[TIME].iloc[0]
        tx_time_s0 = tx_s0_filtered[TIME].iloc[0]
        rx_time_s1 = rx_s1_filtered[TIME].iloc[0]

        if rx_time_s0 < tx_time_s0:  
          shiftLatency = rx_time_s1 - rx_time_s0
          shiftClocks = tx_time_s0 - rx_time_s0
        else:
          shiftLatency = -(rx_time_s1 - rx_time_s0)
          shiftClocks = rx_time_s0 - tx_time_s0

        print( tx_time_s0 )
        print( rx_time_s0 )
        print( shiftClocks )

    if pandas.notna(shiftClocks) and pandas.notna(shiftLatency):
      data.loc[data[ROLE] == "rx", TIME] = data.loc[data[ROLE] == "rx", TIME] + shiftClocks + shiftLatency
      updateData[name] = data
      flag = True

  if True == flag:
    return updateData

  return separatedData

if __name__ == "__main__":
  arguments = argparse.ArgumentParser( )
  arguments.add_argument("-p", "--input", type=str, required=True, help="Path to input directory")
  arguments.add_argument("-d", "--output", type=str, required=True, help="Path to output directory")
  arguments.add_argument("-t", "--transmitter", type=str, required=True, help="Transmitter node number")  
  arguments.add_argument("-r", "--receiver", type=str, required=True, help="Receiver node number")  
  args = arguments.parse_args( )

  txFiles = findFiles( args.input, args.transmitter, "tx" )  
  rxFiles = findFiles( args.input, args.receiver, "rx" )  

  allFiles = txFiles + rxFiles

  commsDataFrames = [ ]
  noiseDataFrames = [ ]

  for path in allFiles:
    fldf = readFile( path )    
    if fldf is not None:
      if 'noise' == parseFileName(path)['type']:
        noiseDataFrames.append(fldf)
      else:
        commsDataFrames.append(fldf)
    else:
      print(f"Skipping file {path}: unable to parse filename.")
    
  commsTable = pandas.concat(commsDataFrames, ignore_index=True) if commsDataFrames else pandas.DataFrame(columns=PD_HEADERS_1)
  noiseTable = pandas.concat(noiseDataFrames, ignore_index=True) if noiseDataFrames else pandas.DataFrame(columns=PD_HEADERS_2)

  print("\nData processing complete.")
  if not commsTable.empty:
    commsTable = correctCoordinates( commsTable )
    commsTable = calculateDistance( commsTable )

    print(f"Comms DataFrame successfully created with {len(commsTable)} entries.")

    exportDf = separateDataFrames( commsTable )
    exportDf = timeShift( exportDf )

    for name, data in exportDf.items( ):
      savePath = os.path.join( args.output, f"{name}.csv" )
      data.to_csv(savePath, index=False)

  if not noiseTable.empty:
    print(f"Comms DataFrame successfully created with {len(noiseTable)} entries.")
