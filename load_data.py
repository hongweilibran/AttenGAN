import os
import numpy as np
from PIL import Image
from keras.utils import Sequence
#from skimage.io import imread


def load_data(nr_of_channels, batch_size=1, nr_T1_train_imgs=None, nr_T2_train_imgs=None, nr_S1_train_imgs=None, nr_S2_train_imgs=None, nr_train_mask_imgs=None, nr_T1_test_imgs=None, nr_T2_test_imgs=None, nr_S1_test_imgs=None, nr_S2_test_imgs=None, nr_test_mask_imgs=None, subfolder='',
              generator=False, D_model=None, use_multiscale_discriminator=False, use_supervised_learning=False, REAL_LABEL=1.0):
    data_path = 'data/MR'

    trainT1_path = os.path.join(data_path, subfolder, 'trainT1')
    trainT2_path = os.path.join(data_path, subfolder, 'trainT2')
    trainS1_path = os.path.join(data_path, subfolder, 'trainS1')
    trainS2_path = os.path.join(data_path, subfolder, 'trainS2')
    train_mask_path = os.path.join(data_path, subfolder, 'train_mask')

    testT1_path = os.path.join(data_path, subfolder, 'testT1')
    testT2_path = os.path.join(data_path, subfolder, 'testT2')
    testS1_path = os.path.join(data_path, subfolder, 'testS1')
    testS2_path = os.path.join(data_path, subfolder, 'testS2')
    test_mask_path = os.path.join(data_path, subfolder, 'test_mask')
    
    trainT1_image_names = os.listdir(trainT1_path)
    if nr_T1_train_imgs != None:
        trainT1_image_names = trainT1_image_names[:nr_T1_train_imgs]
    trainT2_image_names = os.listdir(trainT2_path)
    if nr_T2_train_imgs != None:
        trainT2_image_names = trainT2_image_names[:nr_T2_train_imgs]
    trainS1_image_names = os.listdir(trainS1_path)
    if nr_S1_train_imgs != None:
        trainS1_image_names = trainS1_image_names[:nr_S1_train_imgs]
    trainS2_image_names = os.listdir(trainS2_path)
    if nr_S2_train_imgs != None:
        trainS2_image_names = trainS2_image_names[:nr_S2_train_imgs] 
    train_mask_image_names = os.listdir(train_mask_path)
    if nr_train_mask_imgs != None:
        train_mask_image_names = train_mask_image_names[:nr_train_mask_imgs] 

    testT1_image_names = os.listdir(testT1_path)
    testT2_image_names = os.listdir(testT2_path)
    
    if nr_T1_test_imgs != None:
        testT1_image_names = testT1_image_names[:nr_T1_test_imgs]
    if nr_T2_test_imgs != None:
        testT2_image_names = testT2_image_names[:nr_T2_test_imgs]

    testS1_image_names = os.listdir(testS1_path)
    if nr_S1_test_imgs != None:
        testS1_image_names = testS1_image_names[:nr_S1_test_imgs]
    testS2_image_names = os.listdir(testS2_path)
    if nr_S2_test_imgs != None:
        testS2_image_names = testS2_image_names[:nr_S2_test_imgs]
    test_mask_image_names = os.listdir(test_mask_path)
    if nr_test_mask_imgs != None:
        test_mask_image_names = test_mask_image_names[:nr_test_mask_imgs] 

    if generator:
        return data_sequence(trainT1_path, trainT2_path, trainS1_path, trainS2_path, train_mask_path, trainT1_image_names, trainT2_image_names, trainS1_image_names, trainS2_image_names, test_mask_path, batch_size=batch_size)  # D_model, use_multiscale_discriminator, use_supervised_learning, REAL_LABEL)
    else:
        trainT1_images = create_image_array(trainT1_image_names, trainT1_path, nr_of_channels)
        trainT2_images = create_image_array(trainT2_image_names, trainT2_path, nr_of_channels)
        trainS1_images = create_image_array(trainS1_image_names, trainS1_path, nr_of_channels)
        trainS2_images = create_image_array(trainS2_image_names, trainS2_path, nr_of_channels)
        train_mask_images = create_image_array(train_mask_image_names, train_mask_path, nr_of_channels)
        testT1_images = create_image_array(testT1_image_names, testT1_path, nr_of_channels)
        testT2_images = create_image_array(testT2_image_names, testT2_path, nr_of_channels)
        testS1_images = create_image_array(testS1_image_names, testS1_path, nr_of_channels)
        testS2_images = create_image_array(testS2_image_names, testS2_path, nr_of_channels)
        test_mask_images = create_image_array(test_mask_image_names, test_mask_path, nr_of_channels)
        return {"trainT1_images": trainT1_images, 
                "trainT2_images": trainT2_images, 
                "trainS1_images": trainS1_images, 
                "trainS2_images": trainS2_images,
                "train_mask_images": train_mask_images,
                "testT1_images": testT1_images, 
                "testT2_images": testT2_images, 
                "testS1_images": testS1_images, 
                "testS2_images": testS2_images,
                "test_mask_images": test_mask_images,
                "trainT1_image_names": trainT1_image_names,
                "trainT2_image_names": trainT2_image_names,
                "trainS1_image_names": trainS1_image_names,
                "trainS2_image_names": trainS2_image_names,
                "train_mask_image_names": train_mask_image_names,
                "testT1_image_names": testT1_image_names,
                "testT2_image_names": testT2_image_names,
                "testS1_image_names": testS1_image_names,
                "testS2_image_names": testS2_image_names,
                "test_mask_image_names": test_mask_image_names}
                

def create_image_array(image_list, image_path, nr_of_channels):
    image_array = []
    for image_name in image_list:
        if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
            if nr_of_channels == 1:  # Gray scale image -> MR image
                image = np.array(Image.open(os.path.join(image_path, image_name)))
                image = image[:, :, np.newaxis]
            else:                   # RGB image -> 3 channels
                image = np.array(Image.open(os.path.join(image_path, image_name)))
            image = normalize_array(image)
            image_array.append(image)

    return np.array(image_array)
  
  
  # If using 16 bit depth images, use the formula 'array = array / 32767.5 - 1' instead
def normalize_array(array):
    array = array / 127.5 - 1
 #   array = array / 100
    return array


class data_sequence(Sequence):

    def __init__(self, trainT1_path, trainT2_path, trainS1_path, trainS2_path, train_mask_path, test_mask_path, image_list_T1, image_list_T2, image_list_S1, image_list_S2, image_list_mask, batch_size=1):  # , D_model, use_multiscale_discriminator, use_supervised_learning, REAL_LABEL):
        self.batch_size = batch_size
        self.train_T1 = []
        self.train_T2 = []
        self.train_S1 = []
        self.train_S2 = []
        self.train_mask = []
        self.train_mask = []
        for image_name in image_list_T1:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_T1.append(os.path.join(trainT1_path, image_name))
        for image_name in image_list_T2:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_T2.append(os.path.join(trainT2_path, image_name))
        for image_name in image_list_S1:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_S1.append(os.path.join(trainS1_path, image_name))
        for image_name in image_list_S2:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_S2.append(os.path.join(trainS2_path, image_name))
        for image_name in image_list_train_mask:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_mask.append(os.path.join(train_mask_path, image_name))
        for image_name in image_list_test_mask:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.test_mask.append(os.path.join(test_mask_path, image_name))

    def __len__(self):
        return int(max(len(self.train_T1), len(self.train_T2), len(self.train_S1), len(self.train_S2)) / float(self.batch_size))

    def __getitem__(self, idx):  # , use_multiscale_discriminator, use_supervised_learning):if loop_index + batch_size >= min_nr_imgs:
#        if idx >= min(len(self.train_A), len(self.train_B)):
#            # If all images soon are used for one domain,
#            # randomly pick from this domain
#            if len(self.train_A) <= len(self.train_B):
#                indexes_A = np.random.randint(len(self.train_A), size=self.batch_size)
#                batch_A = []
#                for i in indexes_A:
#                    batch_A.append(self.train_A[i])
#                batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]
#            else:
#                indexes_B = np.random.randint(len(self.train_B), size=self.batch_size)
#                batch_B = []
#                for i in indexes_B:
#                    batch_B.append(self.train_B[i])
#                batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
#        else:
        batch_T1 = self.train_T1[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_T2 = self.train_T2[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_S1 = self.train_S1[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_S2 = self.train_S2[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_train_mask = self.train_mask[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_test_mask = self.train_mask[idx * self.batch_size:(idx + 1) * self.batch_size]

        real_images_T1 = create_image_array(batch_T1, '', 3)
        real_images_T2 = create_image_array(batch_T2, '', 3)
        real_images_S1 = create_image_array(batch_S1, '', 3)
        real_images_S2 = create_image_array(batch_S2, '', 3)
        real_images_train_mask = create_image_array(batch_train_mask, '', 3)
        real_images_test_mask = create_image_array(batch_test_mask, '', 3)
        return real_images_T1, real_images_T2, real_images_S1, real_images_S2, real_images_train_mask, real_images_train_mask  # input_data, target_data


if __name__ == '__main__':
    load_data()
