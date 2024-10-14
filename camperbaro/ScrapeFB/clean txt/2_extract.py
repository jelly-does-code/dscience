import pandas as pd
import re

def process_text_file(file_path):
    objects = []
    current_object = {}
    object_index = 1

    # Expanded list of car brands with an emphasis on campervans
    car_brands = ["toyota", "ford", "chevrolet", "honda", "bmw", "mercedes", "audi", "volkswagen", "tesla", "nissan", "subaru", "chrysler", "fiat", "dodge", "mazda", "jeep", "kia", "hyundai", "volvo", "lexus", "jaguar", "land rover", "porsche", "mitsubishi", "mini", "buick", "cadillac", "lincoln", "infiniti", "acura", "alfa romeo", "genesis", "ram", "gmc", "fiat", "smart", "maserati", "bentley", "bugatti", "ferrari", "lamborghini", "lotus", "maybach", "morgan", "rolls-royce", "aston martin", "subaru", "skoda", "seat", "peugeot", "renault", "citroen", "opel", "vauxhall", "fiat", "abarth", "lancia", "dacia", "suzuki", "isuzu", "mahindra", "tata", "daihatsu", "mclaren", "koenigsegg", "alpina", "brabus", "koopman", "argo", "ram", "california", "westfalia", "winnebago", "pleasure-way", "airstream", "coachmen", "sportsmobile", "chinook"]

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        for line in lines:
            # Check for the start of a new object description
            if line.strip() == "________________________":
                if current_object:
                    current_object['index'] = object_index
                    objects.append(current_object)
                    object_index += 1
                current_object = {}
            else:
                # Check for the presence of the '$' symbol
                if '$' in line:
                    # Extract the string following the dollar sign (space separated)
                    price_string = line.split('$')[1].split('·')[0].strip()
                    
                    # Remove non-numeric characters from the 'price' string
                    current_object['price'] = re.sub(r'[^0-9.]', '', price_string)

                    # Extract the last and second to last strings as 'state' and 'city'
                    words = line.split()
                    if len(words) >= 2:
                        current_object['state'] = words[-1]
                        
                        # Extract the string between · symbol and , symbol as 'city'
                        city_match = re.search(r'·(.*?),', line)
                        if city_match:
                            current_object['city'] = city_match.group(1).strip()

                # Check for the presence of the '*' symbols
                if '*' in line:
                    # Extract the first string between '*' characters as 'seller_first_name'
                    seller_match = re.search(r'\*([^*]+)\*', line)
                    if seller_match:
                        current_object['seller_first_name'] = seller_match.group(1).strip()

                # Check for the line starting with four digits and a space, and containing exactly 3 strings
                year_brand_model_match = re.match(r'^(\d{4})\s+(\S+)\s+(\S+)', line)
                if year_brand_model_match:
                    current_object['year_built'] = int(year_brand_model_match.group(1))
                    brand = year_brand_model_match.group(2)
                    model = year_brand_model_match.group(3).strip()

                    # Check for the presence of a car brand
                    if any(brand in car_brand for car_brand in car_brands):
                        current_object['brand'] = brand

                        # Extract the string after the 'brand' property as 'model'
                        current_object['model'] = model

                # Check for the presence of "solar" or "no solar"
                if "solar" in line and "no solar" not in line:
                    current_object['solar'] = 'y'
                else:
                    current_object['solar'] = 'n'

                # Check for the presence of "odo" or "km/kms" and extract the number before it as 'odo'
                odo_match = re.search(r'(\d+|\d+,\d+)\s*(odo|km|kms)', line)
                if odo_match:
                    # Replace "xxx" with "500" in the extracted number
                    odo_value = re.sub(r'xxx', '500', odo_match.group(1))
                    current_object['odo'] = int(odo_value.replace(',', ''))

                # Check for the presence of "petrol" or "diesel" and assign value to 'fuel'
                if "petrol" in line:
                    current_object['fuel'] = 'petrol'
                elif "diesel" in line:
                    current_object['fuel'] = 'diesel'

                # You can add more conditions to extract other attributes if needed

    # Create a DataFrame from the list of objects
    df = pd.DataFrame(objects)
    return df

if __name__ == "__main__":
    file_path = "export_cleaned.txt"
    result_df = process_text_file(file_path)

    # Print the resulting DataFrame with an index column
    print(result_df)
