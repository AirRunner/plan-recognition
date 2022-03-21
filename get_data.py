from os import remove
from wget import download
from zipfile import ZipFile


# Download
download("https://airrunner-public.s3.ca-central-1.amazonaws.com/plan_recognition_data/data_part1.zip")
download("https://airrunner-public.s3.ca-central-1.amazonaws.com/plan_recognition_data/data_part2.zip")


# Extract data
print("\n\nExtracting data...")
with ZipFile("data_part1.zip") as zip:
  zip.extractall(path="part1/")
remove("data_part1.zip")

with ZipFile("data_part2.zip") as zip:
  zip.extractall(path="part2/")
remove("data_part2.zip")


print("-- Done --")
