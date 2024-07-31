import sys
import numpy as np
import pandas as pd

class Logger:
    def __init__(self,file="logs/general.log",method="a"):
        self.file = file
        self.method = method
        if method == "w":
            with open(file,"w") as logs:
                pass
    
    def log(self,data):
        with open(self.file,"a") as logs:
            sys.stdout = logs
            if type(data) == pd.DataFrame:
                with pd.option_context("display.max_rows",None,
                                    'display.max_columns',None,
                                    'display.width',None,
                                    'display.max_colwidth',None):
                    print(data)
            elif isinstance(data, np.ndarray):
                with np.printoptions(threshold=np.inf, linewidth=np.inf):
                        print(data)
            else:
                print(data)
            sys.stdout = sys.__stdout__
          

