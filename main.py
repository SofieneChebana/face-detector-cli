import sys
from script import start_analysis
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image retrieval tool.")
    parser.add_argument(
      "-f",
      "--folder",
      type=Path,
      required=True,
      help="Path to the folder to analyze.",
  )
    parser.add_argument(
          "-i",
          "--target",
          type=Path,
          required=True,
          help="Path to the target image.",
      )

    args = parser.parse_args()

    if args.folder is None :
        parser.error("Folder not specified.")

    if args.target is None :
            parser.error("Target not specified")   

    source_dir  = Path(args.folder)
    target      = Path(args.target)

    if not source_dir.exists() or not source_dir.is_dir():
        parser.error(f"Source folder does not exist or is not a folder: {source_dir}")

    if not target.exists() or target.is_dir():
        parser.error(f"Target does not exist or is not a file: {target}")
        
    start_analysis(source_dir, target)