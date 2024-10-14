import csv
import random
num_records = 1000
brands = ["Volkswagen Westfalia", "Ford Transit", "Mercedes-Benz Sprinter", "Fiat Ducato", "Chevrolet Express", "Volkswagen California", "Dodge Ram Promaster", "Hymer Aktiv", "Nissan NV200", "Toyota Hiace"]
years = [x for x in range(2000, 2025)]
currencies = ["AUD"]
countries = ["Australia"]
zip_codes = [
    2000,  # Sydney, NSW
    3000,  # Melbourne, VIC
    4000,  # Brisbane, QLD
    6000,  # Perth, WA
    5000,  # Adelaide, SA
    4217,  # Gold Coast, QLD
    2300,  # Newcastle, NSW
    2600,  # Canberra, ACT
    4551,  # Sunshine Coast, QLD
    2500,  # Wollongong, NSW
    3220,  # Geelong, VIC
    7000,  # Hobart, TAS
    4810,  # Townsville, QLD
    4870,  # Cairns, QLD
    4350,  # Toowoomba, QLD
    3350,  # Ballarat, VIC
    3550,  # Bendigo, VIC
    7250,  # Launceston, TAS
    2640,  # Albury-Wodonga, NSW/VIC
    4740,  # Mackay, QLD
    4700,  # Rockhampton, QLD
    6230,  # Bunbury, WA
    2450,  # Coffs Harbour, NSW
    4670,  # Bundaberg, QLD
    2650,  # Wagga Wagga, NSW
    4655,  # Hervey Bay, QLD
    3500,  # Mildura, VIC
    3630,  # Shepparton-Mooroopna, VIC
    2830   # Dubbo, NSW
    ]
service_history = ["y", "n"]
month_of_year = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

with open("campervans_data.csv", "w", newline="") as csvfile:
    fieldnames = ["brand", "price", "year_made", "currency", "country", "zip_code", "odometer", "service_history", "doy", "moy"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for _ in range(num_records):
        brand = random.choice(brands)
        price = random.randint(10000, 60000)
        year_made = random.randint(2000, 2024)
        currency = 'AUD'
        country = 'australia'
        zip_code = random.choice(zip_codes)
        odometer = random.randint(0, 700000)
        servicing = random.choice(service_history)
        doy = random.randint(1, 365)
        moy = random.randint(1, 13)
        writer.writerow({
            "brand": brand,
            "price": price,
            "year_made": year_made,
            "currency": currency,
            "country": country,
            "zip_code": zip_code,
            "odometer": odometer,
            "service_history": servicing,
            "doy": doy,
            "moy": moy
        })
