import time

import clahe
import os
import cv2

import imageio.v2 as imageio

# directory to the non-enhanced dataset
project_directory = os.path.dirname(os.path.abspath(__file__))

# go to the parent directory
project_directory = os.path.dirname(project_directory)

benchmark_directory = os.path.join(project_directory, "benchmark")
if not os.path.exists(benchmark_directory):
    os.mkdir(benchmark_directory)


dataset_directory = os.path.join(project_directory, "data")

non_enhanced_dataset_directory = os.path.join(dataset_directory, "non-enhanced")
non_enhanced_covid_19_directory = os.path.join(non_enhanced_dataset_directory, "COVID-19")
non_enhanced_non_covid_19_directory = os.path.join(non_enhanced_dataset_directory, "Non-COVID-19")

# create enhanced dataset directory
enhanced_dataset_directory = os.path.join(dataset_directory, "enhanced")
enhanced_covid_19_directory = os.path.join(enhanced_dataset_directory, "COVID-19")
enhanced_non_covid_19_directory = os.path.join(enhanced_dataset_directory, "Non-COVID-19")

if not (os.path.exists(enhanced_dataset_directory) and os.path.exists(enhanced_covid_19_directory)
        and os.path.exists(enhanced_non_covid_19_directory)):
    os.mkdir(enhanced_dataset_directory)
    os.mkdir(enhanced_covid_19_directory)
    os.mkdir(enhanced_non_covid_19_directory)
    print("Created enhanced dataset directory")


# list of images in the non-enhanced dataset
non_enhanced_covid_19_images = os.listdir(non_enhanced_covid_19_directory)
print("Number of COVID-19 images in the non-enhanced dataset: ", len(non_enhanced_covid_19_images))

non_enhanced_non_covid_19_images = os.listdir(non_enhanced_non_covid_19_directory)
print("Number of Non-COVID-19 images in the non-enhanced dataset: ", len(non_enhanced_non_covid_19_images))

start_time = time.time()
# enhance the images
for _image in non_enhanced_covid_19_images:
    image_path = os.path.join(non_enhanced_covid_19_directory, _image)
    image = cv2.imread(image_path, 0)
    threshold_image, enhanced_image = clahe.clahe_opencv(image, 2.0, 8)
    imageio.imwrite(os.path.join(enhanced_covid_19_directory, _image), enhanced_image)

for _image in non_enhanced_non_covid_19_images:
    image_path = os.path.join(non_enhanced_non_covid_19_directory, _image)
    image = cv2.imread(image_path, 0)
    threshold_image, enhanced_image = clahe.clahe_opencv(image, 2.0, 8)
    cv2.imwrite(os.path.join(enhanced_non_covid_19_directory, _image), enhanced_image)

end_time = time.time()

print("Time taken to enhance the dataset: ", end_time - start_time)

#  save the benchmarking results
with open(os.path.join(benchmark_directory, "enhance_benchmark.txt"), "a") as f:
    f.write("Time taken to enhance the dataset: " + str(end_time - start_time) + "\n")





