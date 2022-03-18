import pandas as pd
import numpy as np
import math
from pathlib import Path


def countBelowMidpoint(file: str):
    df = pd.read_csv(file, sep='\t')
    count = df["below wall"].value_counts()
    print(f"{file}:\t {count}")



def getAverageMovement(file: str):
    df = pd.read_csv(file, sep='\t')
    average_movements = []

    for id in pd.unique(df["id"]):
        centroids = df[df["id"] == id][['x','y']].to_numpy(np.float32)
        movements = np.array([np.linalg.norm(centroids[i] - centroids[i+1]) for i in range(len(centroids) - 1)])
        average_movement = np.average(movements)
        
        # Check for nan value which may result in case of collapsed cells
        if not math.isnan(average_movement):
            average_movements.append(np.average(movements))
    return np.average(average_movements)
        
                
def main():
    root = Path("logs")
    results = [] 
    
    for f in  root.rglob("*.csv"):
        results.append([f, getAverageMovement(f)])
        
    df = pd.DataFrame(results, columns=["File", "Avg. movement"])
    print(df)
    
    for f in root.rglob("*wall*.csv"):
        countBelowMidpoint(f)


if __name__ == "__main__":
    main()
