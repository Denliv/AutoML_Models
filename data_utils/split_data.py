import os
import shutil

def split_data(data_dir, output_dir, train_ratio, val_ratio, image_per_class):
    classes = os.listdir(data_dir)

    # Создание директорий
    for split in ["train", "val"]:
        for class_name in classes:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)

        for count, img in enumerate(os.listdir(class_path)):
            if count >= image_per_class:
                break
            if 0 <= count < train_ratio * image_per_class:
                shutil.copy(os.path.join(class_path, img), os.path.join(output_dir, "train", class_name, img))
            else:
                shutil.copy(os.path.join(class_path, img), os.path.join(output_dir, "val", class_name, img))

if __name__ == "__main__":
    split_data("..\\data", "..\\.yolo_data", 0.8, 0.2, 3000)