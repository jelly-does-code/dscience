# This is the first module that will execute
# It deals with the raw, unprocessed manual export from Facebook Marketplace groups.

import re

def remove_icons_from_line(line):
        icon_pattern = re.compile(r'[\W_]+')
        return icon_pattern.sub('', line)

def clean_raw_export(input_file, output_file):

    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Find the index of the line starting with "<https://www.facebook.com/groups/525139527623724/posts/2750884801715841/"
    # All lines before that are not interesting
    start_index = 0
    for i, line in enumerate(lines):
        if line.startswith("<https://www.facebook.com/groups/525139527623724/posts/2750884801715841/"):
            start_index = i
            break

    # Remove lines until the line starting with the specified URL
    cleaned_lines = lines[start_index + 1:]

    # Make everything lowercase
    cleaned_lines = [line.lower() for line in cleaned_lines]

    # Remove lines starting with '<'
    cleaned_lines = [line for line in cleaned_lines if not line.startswith('<')]

    # Remove lines starting with "Chatbericht sturen"
    cleaned_lines = [line for line in cleaned_lines if not line.startswith('Chatbericht sturen')]

    # Remove lines starting with "Alle reacties:"
    cleaned_lines = [line for line in cleaned_lines if not line.startswith('Alle reacties:')]

    # Remove lines starting with "Opmerking plaatsen"
    cleaned_lines = [line for line in cleaned_lines if not line.startswith('Opmerking plaatsen')]

    # Remove lines starting with "Meer opmerkingen weergeven"
    cleaned_lines = [line for line in cleaned_lines if not line.startswith('Meer opmerkingen weergeven')]

    # Remove lines starting with <https://www.facebook.com/groups/ once spaces are removed
    cleaned_lines = [line for line in cleaned_lines if not line.replace(" ", "").startswith('<https://www.facebook.com/groups/')]

    # Remove lines starting with "Beantwoorden" once spaces are removed
    cleaned_lines = [line for line in cleaned_lines if not line.replace(" ", "").startswith('Beantwoorden')]

    # Replace "Gedeeld met Openbare groep" with "________________________"
    cleaned_lines = [line.replace("Gedeeld met Openbare groep", "________________________") for line in cleaned_lines]

    # Remove lines containing "Schrijf een openbare opmerking"
    cleaned_lines = [line for line in cleaned_lines if "Schrijf een openbare opmerking" not in line]

    # Remove lines containing "Populaire advertenties"
    cleaned_lines = [line for line in cleaned_lines if "Populaire advertenties" not in line]

    # Remove lines containing "groepsoverzicht sorteren op"
    cleaned_lines = [line for line in cleaned_lines if "groepsoverzicht sorteren op" not in line]

    # Remove lines with less than 10 characters when empty space is removed
    cleaned_lines = [line for line in cleaned_lines if len(line.strip()) >= 10]

    with open("tmp.txt", 'w', encoding='utf-8') as output:
        output.writelines(cleaned_lines)
    
    '''
    # Make everything lowercase
    with open("tmp.txt", 'r', encoding='utf-8') as input_file:
        content = input_file.read()
        lowercase_content = content.lower()
    with open(output_file, 'w', encoding='utf-8') as output_file:
        output_file.write(lowercase_content)
    '''

if __name__ == "__main__":
    clean_raw_export("raw_export.txt", "phase1_export.txt")
