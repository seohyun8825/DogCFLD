import os
import shutil

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def format_path(path):

    path = path.lower().replace('\\', '/')

    path = path.replace('/', '')

    path = path.replace('_', '')
    return path

def make_dataset(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    train_root = os.path.join(dir, 'train_highres')
    if not os.path.exists(train_root):
        os.makedirs(train_root)

    test_root = os.path.join(dir, 'test_highres')
    if not os.path.exists(test_root):
        os.makedirs(test_root)

    # 파일명을 키로, 원본 파일명을 값으로 저장
    train_images = {}
    with open(os.path.join(dir, 'train.lst'), 'r') as train_f:
        for line in train_f:
            original_line = line.strip()
            normalized_line = format_path(original_line)
            train_images[normalized_line] = original_line

    test_images = {}
    with open(os.path.join(dir, 'test.lst'), 'r') as test_f:
        for line in test_f:
            original_line = line.strip()
            normalized_line = format_path(original_line)
            test_images[normalized_line] = original_line

    for root, _, fnames in os.walk(dir):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                formatted_path = format_path(path)
                if formatted_path in train_images:
                    target_fname = train_images[formatted_path]
                    shutil.copy(path, os.path.join(train_root, target_fname))
                    print("Copying to train: ", target_fname)
                if formatted_path in test_images:
                    target_fname = test_images[formatted_path]
                    shutil.copy(path, os.path.join(test_root, target_fname))
                    print("Copying to test: ", target_fname)

make_dataset('fashion')
