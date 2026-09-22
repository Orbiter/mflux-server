
import json
import os
import random

def read_json_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def construct_filename(examples):
    return "_".join(example.replace(" ", "-").replace("(", "").replace(")", "").lower() for example in examples)

def construct_prompt(categories, examples):
    return ", ".join(f"{category['name']}: {example}" for category, example in zip(categories, examples))

def generate_prompt_files(data, output_filename, num_entries=100):
    categories = data.get('categories', [])
    num_categories = len(categories)
    
    with open(output_filename, 'w') as outfile:
        for _ in range(num_entries):
            # for each category, there is a category boost value from 1 to 9
            # the higher the value, the more likely the category will be selected
            # category['boost'] is the boost value
            preselected_categories = [] # the selected categories
            for category in categories:
                if random.random() < category['boost'] / 10:
                    preselected_categories.append(category)

            # Randomly select 3 categories and 1 example per category
            selected_categories = random.sample(preselected_categories, 3)

            # Select 1 example from each category, but prefer examples from the beginning of the list
            # A likelihood shall be calculated based on the index of the example in the list
            # The likelihood is calculated as 1 / (index + 1)
            # The example with the highest likelihood will be selected
            selected_examples = []
            for category in selected_categories:
                examples = category['examples']
                likelihoods = [1 / (index + 1) for index in range(len(examples))]
                # randomly select three examples from the category with likelihood
                selected_example = random.choices(examples, weights=likelihoods, k=1)[0]
                selected_examples.append(selected_example)
            
            # Construct the prompt and filename
            prompt = construct_prompt(selected_categories, selected_examples)
            tags = construct_filename(selected_examples)
            
            result = {
                "prompt": prompt,
                "tags": tags
            }
            
            # Write each JSON object on a new line
            json.dump(result, outfile)
            outfile.write('\n')  # Ensure each JSON object is on a new line


def main():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_directory, 'factory-categories.json')
    output_file_path = os.path.join(current_directory, 'factory-prompts.jsonlist')
    
    # Read the JSON file
    data = read_json_file(json_file_path)
    
    # Generate prompt files
    generate_prompt_files(data, output_file_path, num_entries=100000)

if __name__ == '__main__':
    main()
