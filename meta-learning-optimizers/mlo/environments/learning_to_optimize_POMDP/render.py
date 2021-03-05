from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np


class RenderView():

    def __init__(self, output_path):

        self.output_path = output_path
        self.fig, self.ax = plt.subplots()
        self.task_idx = 0
        self.a = 0
        self.t = 0

        self.bounds = [0, 1]

    def reset(self):
        self.t = 0
        self.task_idx += 1
        plt.close('all')
        self.fig, self.ax = plt.subplots()

    def save(self):

        plt.axis('off')
        plt.rc('axes', labelsize=8)
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=8)
        plt.rc('figure', figsize=[2.5, 2.5])


        plt.xlim(self.bounds[0], self.bounds[1])
        plt.ylim(self.bounds[0], self.bounds[1])
        plt.savefig(f'{self.output_path}/T{self.task_idx}_{self.t}.png', transparent = False, bbox_inches = 'tight', pad_inches=0)
        #img = self.fig2img()
        #img.save(f'{self.output_path}/T{self.task_idx}_{self.t}.png')

        self.t += 1
        return self

    def fig2img(self):
        r'''
        Description: 
            - Convert a Matplotlib figure to a 4D numpy array 
              with RGBA channels and return it
        Input: 
            - fig: matplotlib figure
        Output:
            - numpy 3D array of RGBA values
        '''

        # Draw the renderer
        self.fig.canvas.draw()

        # Get the RGBA buffer from the figure
        w, h = self.fig.canvas.get_width_height()
        buff = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
        buff.shape = (w, h, 4)

        # Roll the ALPHA channel to have it in RGBA mode
        buff = np.roll(buff, 3, axis=2)

        # Convert numpy array to PIL Image
        w, h, d = buff.shape
        return Image.frombytes("RGBA", (w, h), buff.tobytes())

    def build(self, task):

        # Render Function:
        temp_task = deepcopy(task)
        self.title = temp_task.__str__()

        # Set xs and ys:
        resolution = 500
        self.x1, self.x2 = np.meshgrid(np.linspace(self.bounds[0], self.bounds[1], resolution), np.linspace(self.bounds[0], self.bounds[1], resolution))
        X = np.array([self.x1.flatten(), self.x2.flatten()]).T

        try:
            self.Y = np.load(open('AAAAAAAAAAAAA.npy', 'rb'))

        except:
            add = []
            for i in range(len(X)):
                print(i, len(X))
                add.append(temp_task(X[i]))
            self.Y = np.array(add).reshape(resolution, resolution)
            np.save(open('AAAAAAAAAAAAA.npy', 'wb'), self.Y)
            print("saved!")

        #minY = self.Y.reshape(resolution*resolution)
        #idx_min = minY.argmin()
        #self.xmin, self.ymin = X[idx_min]
        self.xmin, self.ymin = 0.7142857142857142, 0.03
        #self.ymin += 0.02
        #print(self.xmin, self.ymin)
        #exit()
        #print(self.xmin, self.ymin)
        #exit()

        #self.ax.plot(self.xmin, self.ymin,'ro', 'yellow') 
        #self.ax.scatter(self.xmin, self.ymin, s=10, c='y')

        from matplotlib.colors import ListedColormap, LinearSegmentedColormap


        #viridis = plt.cm.get_cmap('viridis', 256)
        #newcolors = viridis(np.linspace(0, 1, 256))
        
        #newcmp = ListedColormap(newcolors)
        color1 = plt.cm.get_cmap('bwr', 512)
        color2 = plt.cm.get_cmap('bwr', 512)
        a = color1(np.linspace(0.0, 1.0, 512))
        b = color2(np.linspace(0.0, 0.2, 512))
        newcolor = a#np.concatenate((a,b))

        newcmp = ListedColormap(newcolor)

        # Plot function:
        self.ax.pcolormesh(self.x1, self.x2, self.Y, cmap=plt.cm.coolwarm, shading='auto')

        del temp_task
        return self

    def add(self, x, clean=True):

        # Clean and plot function:
        if clean == True:
            plt.close('all')
            self.fig, self.ax = plt.subplots()
            self.ax.pcolormesh(self.x1, self.x2, self.Y, cmap=plt.cm.coolwarm, shading='auto')

        self.ax.scatter(np.array([self.xmin]).T, np.array([self.ymin]).T, s=100, c='yellow', marker='*')
        self.ax.scatter(x[:, 0], x[:, 1], s=60, c='black', marker='X')

        #x[:, 1] += 0.02
        return self