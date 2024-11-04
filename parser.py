# This script only converts a combined level files into individual text files
import os
import sys

def parse_all_levels(input_file):
    output_dir = "puzzles/puzzle_set_" + input_file[-7:-4]
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "r") as infile:
        content = infile.read()

    levels = content.split(";")

    for level in levels:
        lines = level.strip().splitlines()
        if lines:
            level_number = int(lines[0].strip())
            level_content = "\n".join(lines[1:])
            with open(
                os.path.join(output_dir, f"level_{level_number}.txt"), "w"
            ) as outfile:
                outfile.write(level_content)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parser.py puzzles/NNN.txt")
        sys.exit(1)

    parse_all_levels(sys.argv[1])
