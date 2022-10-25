import numpy as np
import imageio
import matplotlib.pyplot as plt
import math
import shutil


def plot(cirList,h,w):
    #Plots all circles within h x w box

    plt.figure()
    for cir in cirList:
        circle1 = plt.Circle((cir[0],cir[1]),cir[2], facecolor='tab:brown',edgecolor='k')
        plt.gcf().gca().add_artist(circle1)
    plt.xlim(0, w)
    plt.ylim(0, h)
    plt.gca().set_aspect('equal','box')
    plt.show()


def plot_save(cirList,index,filenames,direc,w,h):
    # Creates a plot of circles with information in cirList within h x w box
    # It appends the filename in filenames
    
    plt.figure()
    for cir in cirList:
        circle1 = plt.Circle((cir[0],cir[1]),cir[2], facecolor='tab:brown',edgecolor='k')
        plt.gcf().gca().add_artist(circle1)
    plt.xlim(0, w)
    plt.ylim(0, h)
    plt.gca().set_aspect('equal','box')
    filename = f'images/circle_{index}_{direc}.png'
    filenames.append(filename)
    plt.savefig(filename,dpi=96)
    plt.close()
    return filenames


def create_gif(filenames,horizontal):
    # It creates a gif from all files in filenames

    # Naming of the gif
    if horizontal:
        direc = "Horizontal"
    else:
        direc = "Vertical"
    # File path
    import os
    dirname = os.path.dirname(__file__)
    gif_path = os.path.join(dirname, f'gif/Compaction_Algorithm_{direc}.gif')
      
    # Create gif with imageio
    with imageio.get_writer(gif_path, mode="I") as writer:
        for filename in filenames:
            image = imageio.v2.imread(filename)
            writer.append_data(image)


def delete_img_dir():
    # It delets the content from the images folder
    shutil.rmtree('images', ignore_errors=True)


def calc_void(cirList):
    #Calculates void from current circle list

    temp_void = 1
    #new height/width of the ballast, round up so we include every circle
    z_max = math.ceil(np.amax(cirList[:,1])*100)/100
    x_max = math.ceil(np.amax(cirList[:,0])*100)/100

    for cir in cirList:
        temp_void = temp_void - math.pi*cir[2]**2/(z_max*x_max)

    return round(temp_void,4)


def discretization_domain(cirList,h,w,smallest_d,v,horizontal):
    # It discretizes the 2D specimen space (h x w) into rectangular domains
    # Thickness of this domain needs to be smaller than smallest radius
    # v = void (Important so domain does not get unnecessarily large)

    thickness = smallest_d/2 - 0.001
    cirList_np = np.array(cirList)

    if not horizontal:
        # Vertical discretization
        num_domains = math.floor(h/thickness)
        max_agg_line = math.floor(w/(smallest_d*(1/(1-v))))
        thickness = round(h/num_domains,3) 

        discret_domains = np.zeros([num_domains,max_agg_line,3]) #3 information per circle (x,y,r)

        #Put all circles in their corresponding domain
        for j,t in enumerate(discret_domains):
            condensed_list = cirList_np[np.where((cirList_np[:,1]<=j*thickness+thickness) & (cirList_np[:,1]>=j*thickness))]
            for i,s in enumerate(condensed_list):
                t[i,:]=s 

    else:
        # Horizontal discretization
        num_domains = math.floor(w/thickness)
        max_agg_line = math.floor(h/(smallest_d*(1/(1-v))))
        thickness = round(w/num_domains,3) 

        discret_domains = np.zeros([num_domains,max_agg_line,3]) #3 information per circle (x,y,r)

        #Swap coordinates so in compaction algorithm it can work with the second coordinate
        cirList_np[:,[1,0]] = cirList_np[:,[0,1]]

        #Put all circles in their corresponding domain
        for j,t in enumerate(discret_domains):
            condensed_list = cirList_np[np.where((cirList_np[:,1]<=j*thickness+thickness) & (cirList_np[:,1]>=j*thickness))]
            for i,s in enumerate(condensed_list):
                t[i,:]=s 
   
    return discret_domains, cirList_np, num_domains


def compaction(domain_list,cirList,num_domains,horizontal,save_gif,w,h):
    #Compaction algorithm that takes a discretized domain of circles
    #It compacts along its second axis

    #Empty working copy of discrete domains
    comp_circles = np.zeros(cirList.shape)

    #store snapshots of algorithm
    filenames = []

    #Helper arrays for creating the snapshots
    discret_domain_copy = np.zeros(domain_list.shape)
    np.copyto(discret_domain_copy,domain_list)
    discret_domain_flattend = np.reshape(discret_domain_copy,[num_domains*domain_list.shape[1],domain_list.shape[2]])
    nonzero_domain_flattend = discret_domain_flattend[discret_domain_flattend.any(axis=1)]

    for i in range(domain_list.shape[0]):
        domList1 = domain_list[i]
        #All circles in current and below domains, reshaped so it's a vector of all circles     
        circles_below = np.reshape(domain_list[0:i+1,:,:],[(i+1)*domain_list.shape[1],domain_list.shape[2]])
        
        #All circles in current domain, filtered so there are no zeros
        nonzero_cir_cur = domList1[domList1.any(axis=1)]
        num_cir_cur = nonzero_cir_cur.size//3
        #All circles in current domain, filtered so there are no zeros
        nonzero_cir_below = circles_below[circles_below.any(axis=1)]
        num_cir_below = nonzero_cir_below.size//3

        #Working copy of domain
        comp_cir_domain_cur = nonzero_cir_cur

        for idx in range(nonzero_cir_cur.shape[0]):
            #Current circle selected
            cir1 = nonzero_cir_cur[idx]
            #only circles in x distance r_max are relevant and only the ones that are below (y-coordinate)
            relevant_cir_indizes = np.where((nonzero_cir_below[:,0]-cir1[0] <= cir1[2]+nonzero_cir_below[:,2]) & (nonzero_cir_below[:,0]- cir1[0] >= -(cir1[2]+nonzero_cir_below[:,2])) & (nonzero_cir_below[:,1] < cir1[1]))[0]
            proximity_cir_below = nonzero_cir_below[relevant_cir_indizes]
            if proximity_cir_below.size == 0:
                #it's the bottom one, so move it to the bottom
                comp_cir_domain_cur[idx][1] = cir1[2]
            else:
                #Find maximum y coordinate (minimum travel distance) to all circles below that are within proximity
                comp_cir_domain_cur[idx,1] = np.amax(np.sqrt((cir1[2]+proximity_cir_below[:,2])**2 - (cir1[0]-proximity_cir_below[:,0])**2) + proximity_cir_below[:,1])
            
        #pass compressed domain into saved state
        domain_list[i,0:num_cir_cur,:] = comp_cir_domain_cur
        comp_circles[num_cir_below-num_cir_cur:num_cir_below,:] = comp_cir_domain_cur
        nonzero_domain_flattend[num_cir_below-num_cir_cur:num_cir_below,:] = comp_cir_domain_cur
        
        if save_gif:
            if horizontal:
                #For the snapshot coordinates have to be swapped
                nonzero_domain_flattend[:,[1,0]] = nonzero_domain_flattend[:,[0,1]]
                filenames = plot_save(nonzero_domain_flattend,i,filenames,"horizontal",h,w)
                nonzero_domain_flattend[:,[1,0]] = nonzero_domain_flattend[:,[0,1]]
            else:
                
                filenames = plot_save(nonzero_domain_flattend,i,filenames,"vertical",h,w)

    #In the end, swap coord back so we have [x,y,r]
    if horizontal:
        comp_circles[:,[1,0]] = comp_circles[:,[0,1]]
        domain_list[:,:,[1,0]] = domain_list[:,:,[0,1]]

    if save_gif:
        create_gif(filenames, horizontal)
        #Delete all images from image folder
        delete_img_dir()
    
    return comp_circles


def run_comp_pattern(cirList,h_box,w_box,smallest_d,cur_void,run_pattern,save_gif,show_plot):
    #It runs a sequence of horizontal and vertical compaction
    #run_pattern has form of [#horiz,#vert,#horiz,#vert,...]
    #save_gif has to have same shape as run_pattern: [True,False,False,False,...]
    #Show_plot enables a printout of the state after each compaction

    for idx,direc in enumerate(run_pattern):
        for re in range(direc):
            if idx % 2 == 0:
                #Vertical compaction for each even index
                cirList = run_compaction(cirList,h_box,w_box,smallest_d,cur_void,False,save_gif[idx])
            else:
                #Horizontal compaction for each odd index
                cirList = run_compaction(cirList,h_box,w_box,smallest_d,cur_void,True,save_gif[idx])
            
            if show_plot:
                plot(cirList,h_box,w_box)
    
    return cirList


def run_compaction(cirList,h_box,w_box,smallest_d,cur_void,horizontal,save_gif):
    # This runs a compaction step
    # For each compaction the domain has to be created and compacted

    discret_domains, cirList_np, num_domains = discretization_domain(cirList,h_box,w_box,smallest_d,cur_void,horizontal)
    comp_circles = compaction(discret_domains,cirList_np,num_domains,horizontal,save_gif,h_box,w_box)

    return comp_circles


