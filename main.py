import sys
from script import start_analysis

if __name__ == "__main__":
    SOURCE_DIR = "images/"
    TARGET = "target.jpg"
    if len(sys.argv)>1:
        TARGET = sys.argv[1]
        
    start_analysis(SOURCE_DIR, TARGET)