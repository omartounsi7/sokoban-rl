def transpose_file(input_filename, output_filename):
    # Read the content from the input file
    with open(input_filename, "r") as infile:
        lines = [line.strip() for line in infile.readlines()]

    # Transpose the lines (invert rows and columns)
    transposed_lines = zip(*lines)

    # Write the transposed content to the output file
    with open(output_filename, "w") as outfile:
        for line in transposed_lines:
            outfile.write("".join(line) + "\n")


# Usage:
input_file = "data/puzzle1.txt"  # Replace with your input file name
output_file = "data/puzzle1-1.txt"  # Replace with your desired output file name
transpose_file(input_file, output_file)
