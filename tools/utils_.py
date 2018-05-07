import os
def select_sample(path='/home/dean/lfw/'):
    humen_list=os.listdir(path)
    train_humen=[]
    for humen in humen_list:
        humen_path=os.path.join(path,humen)
        list_img=os.listdir(humen_path)
        if len(list_img)>20:
            train_humen.append(humen)
    return train_humen

