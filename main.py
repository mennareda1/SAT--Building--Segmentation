import train
import glob
import os 


def main():
       root_dir = r"C:\chatbotApp\tiled_6october_1" #->6october
       #root_dir = r"C:\chatbotApp\tiled_Asema_1" # ->asema

       task = "segment" 
     
#############Training parameters ########################
       train_enabled = True
       input_images = r"C:\menna_data\menna_data\misrsat_clipped\clippedmisrSat_updated.tif"
       #input_images=r"C:\menna_data\6october\Misrsat2_6october.tif"
       #stream = open("config_main.yaml", 'r')
       #config = yaml.load(stream,Loader)
       dirs=["train/images","train/labels","val/images","val/labels"]
       dir_list=[]
       for i in dirs: 
             p = os.path.join(root_dir, task, "dataset", i)
             os.makedirs(p,exist_ok=True)
             dir_list.append(p) 
       if train_enabled :
              images=glob.glob(input_images)
              if task=="segment":
                     print("preparing segmentation tiles ")
                     #train.preparing_train_data_segment(images,dir_list[0],dir_list[1])
                     #train.preparing_val_data(dir_list[0],dir_list[1],dir_list[2],dir_list[3])
                     train.preparing_val_data(r"C:\chatbotApp\filterd6october\images",r"C:\chatbotApp\filterd6october\labels",r"C:\chatbotApp\final\segment\dataset\val\images",r"C:\chatbotApp\final\segment\dataset\val\labels")

              if task=="pose":
                     print("preparing pose tiles ")
                     train.preparing_train_data_pose(images,dir_list[0],dir_list[1])
                     train.preparing_val_data(dir_list[0],dir_list[1],dir_list[2],dir_list[3])
                     #train.preparing_val_data(r"C:\Users\iamme\OneDrive\Desktop\chatbotApp\final\segment\dataset\train\images",r"C:\Users\iamme\OneDrive\Desktop\chatbotApp\final\segment\dataset\train\labels",r"C:\Users\iamme\OneDrive\Desktop\chatbotApp\final\segment\dataset\val\images",r"C:\Users\iamme\OneDrive\Desktop\chatbotApp\final\segment\dataset\val\labels")
              
              if task=="detection":
                     print("preparing detection tiles ")
                     train.preparing_train_data_detection(images,dir_list[0],dir_list[1])
                     train.preparing_val_data(dir_list[0],dir_list[1],dir_list[2],dir_list[3])
             

              if task=="obb":
                     print("preparing obb tiles ")
                     train.preparing_train_data_obb(images,dir_list[0],dir_list[1])
                     train.preparing_val_data(dir_list[0],dir_list[1],dir_list[2],dir_list[3])
          
   
   

if __name__ == '__main__':
       main()