import os
def txt_del(f): 
    lines = f.read()
    lines = lines.split('\n')
    num = len(lines)
    # failcount=0
    dellist=[]
    delnum=0
    for i in range(num):      #每行一张图    
        temp = lines[i]
        filename=temp.split('\\')[-1]
        if filename.startswith("999999"):
            print(filename)
            dellist.append(i)
            delnum+=1
    print(dellist)
    for id in range(delnum):       
        del lines[dellist[delnum-id-1]]
    return lines
            
if __name__ == '__main__':
    txtroot='/home/zhangsn/enhance/datasets/003_heben/001_hebing/003_run_tool/00'
    for root, dirs, files in os.walk(txtroot):
        for file in files:
            if file.endswith('.txt'):
                txt_file = os.path.join(root, file)    
                print(txt_file) 
                with open(txt_file, "r",encoding='utf-8') as f:
                    newlines = txt_del(f)
                with open(txt_file, "w",encoding='utf-8') as f:
                    for ttt in newlines:
                        f.write(ttt)
                        f.write('\r\n')
                    f.close