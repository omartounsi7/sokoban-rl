# This script only converts a combined level files into individual text files
import os


def parse_all_levels(input_file):
    output_dir = "puzzles"
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
    input_file = "puzzles/000.txt"
    parse_all_levels(input_file)
