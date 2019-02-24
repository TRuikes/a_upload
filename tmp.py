from pyglet.gl import *
import time
from scipy import signal
import numpy as np
from scipy.io import wavfile
from scipy import fftpack
import matplotlib.pyplot as plt

settings = {}
settings['nbars'] = 100
settings['fmin'] = 50
settings['fmax'] = 8000
settings['music_file'] = 'test6.wav'
settings['sps'] = 40
settings['sample_length'] = 0.1
settings['delay'] = 0.05


class App(pyglet.window.Window):
    def __init__(self,settings,music_file,*args, **kwargs):

         self.settings = settings
         self.audio = audio(self.settings)
         super().__init__(*args, **kwargs)
         self.set_minimum_size(400,300)
         #glClearColor(0.2, 0.3, 0.2, 1.0)






         self.mBars = musicBars(self.settings['nbars'])

         self.player = pyglet.media.Player()
         self.player.queue(
         pyglet.resource.media(self.settings['music_file'], streaming=False)
         )
         self.playstate = 0




    def on_draw(self):
        self.clear()
        #self.bar1.vertices.draw(GL_TRIANGLES)
        for i in range(0, self.settings['nbars']):
            self.mBars.Bar[i].vertices.draw(GL_TRIANGLES)



    def on_resize(self, width, height):
        glViewport(0,0,width,height)


    def on_key_press(self, symbol, modifiers):
        if symbol == 32:
            if self.playstate == 0:
                self.playstate = 1
                self.player.play()
            elif self.playstate == 1:
                self.playstate = 0
                self.player.pause()








class Bar:
    def __init__(self, x0, width):
        self.maxval = 1
        self.y0 = -0.9
        self.h0 = 0.01
        self.x0 = x0
        self.width = width
        self.vertices = pyglet.graphics.vertex_list_indexed(
        4,
        [0,1,2, 2,3,0],
        ('v3f', [
        self.x0, self.y0, 0,
        self.x0+self.width, self.y0, 0,
        self.x0+self.width, self.y0+self.h0, 0,
        self.x0,self.y0+self.h0,0
        ]),
        ('c3f', [1,1,1, 1,1,1, 1,1,1, 1,1,1])
        )

    def updateHeight(self, heigth):
        heigth = heigth/self.maxval
        self.vertices.colors =  [(1-heigth),0.5,0.8, (1-heigth),0.5,0.8,(1-heigth),0.5,0.8, (1-heigth),0.5,0.8]
        self.vertices.vertices = [
        self.x0, self.y0, 0,
        self.x0+self.width, self.y0, 0,
        self.x0+self.width, self.y0+heigth, 0,
        self.x0,self.y0+heigth,0
        ]




class musicBars:
    def __init__(self, n):
        self.width = 1.8
        self.spacing = 0.2 # space in percentage
        self.number_bars = n
        self.bar_width = (self.width*(1-self.spacing))/self.number_bars
        self.space_width = (self.width*self.spacing)/(self.number_bars-1)

        self.x_offset = -(self.width/2)
        self.Bar = {}
        self.x0_bar = {}
        for i in range(0,n):
            x0 = self.x_offset + (self.bar_width + self.space_width)*i
            self.x0_bar[i] = x0
            self.Bar[i] = Bar(x0, self.bar_width)




class audio:
    def __init__(self, settings):
        self.rate, self.audio = wavfile.read(settings['music_file'])
        self.audio = np.mean(self.audio, axis=1)
        self.settings = settings


    def getFreqs(self, current_time):
        dt = int(np.round( self.settings['sample_length'] *self.rate / 2))
        current_index = int(np.round(current_time*self.rate + self.settings['delay']))
        if (current_index > dt) & (current_index < len(self.audio)-dt):
            snippet = self.audio[ current_index-dt:current_index+dt]
        elif (current_index > dt):
            snippet = self.audio[ current_index-dt:]
        elif (current_index < len(self.audio)-dt):
            snippet = self.audio[ :current_index+dt]


        tax = np.linspace(0,snippet.shape[0]/self.rate,snippet.shape[0])
        N = snippet.shape[0]

        # fft_vals = np.fft.fft(snippet)
        # freqs = np.fft.fftfreq(N, d=1/self.rate)
        # mask = freqs > 0
        # freq_out = freqs[mask]
        # fft_out = 2.0*np.abs(fft_vals/2)[mask]


        fft_vals = np.fft.fft(snippet)[0:int(N/2)]/N
        fft_vals[1:] = 2*fft_vals[1:]
        f_vec = np.abs(fft_vals)
        R_a = ((12194.0**2)*np.power(f_vec,4))/(((np.power(f_vec,2)+20.6**2)*np.sqrt((np.power(f_vec,2)+107.7**2)*(np.power(f_vec,2)+737.9**2))*(np.power(f_vec,2)+12194.0**2)))
        f = self.rate*np.arange( int(N/2))/N


        # fft = np.fft.fft(snippet)
        #
        # f = np.linspace(0, self.rate,N)
        # f = f[:N//2]
        # a = np.abs(fft)[:N//2]*1/N
        #
        # a = a[1:]
        # #f = np.log(f[1:])
        # f = f[1:]
        # a = a*f


        idx = np.linspace(self.settings['fmin'], self.settings['fmax'], self.settings['nbars']+1)

        val = np.empty(self.settings['nbars'])
        prtstr = ''
        for i in range(0, len(idx)-1):
            # val[i] = np.mean(a[(f >= idx[i]) & (f < idx[i+1])])
            # prtstr += '{:3.0f} '.format(val[i])
            val[i] = np.mean(R_a[(f >= idx[i]) & (f < idx[i+1])])
        #print(prtstr)
        print(f[1],f[-1],idx[0],idx[-1], np.max(R_a))
        return val



def updateHeights(inter,APP):
    vals = APP.audio.getFreqs(APP.player.time)
    for i in range(0, APP.settings['nbars']):
        APP.mBars.Bar[i].updateHeight(vals[i])





if __name__ == '__main__':
    file = 'test.wav'
    APP = App(settings,file,600,600,'my windows', resizable=True)
    APP.on_draw()
    pyglet.clock.schedule_interval(updateHeights,1/settings['sps'],APP)
    pyglet.app.run()
