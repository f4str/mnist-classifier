import tkinter as tk
import numpy as np
import network as nn
from PIL import Image

class Recognizer():
	def __init__(self, network):
		self.root = tk.Tk()
		self.root.title('Digit Recognizer')
		self.root.configure(background='black', padx=5)
		self.root.resizable(0, 0)
		
		self.predict_button = tk.Button(self.root, text='Predict', command = self.predict, 
			bg='black', fg='white', width=20, padx=0, bd=4)
		self.predict_button.config(font=('Calibri', 12))
		self.predict_button.grid(row=0)
		
		self.clear_button = tk.Button(self.root, text='Clear', command=self.clear, 
			bg='black', fg='white', width=20, padx=0, bd=4)
		self.clear_button.config(font=('Calibri', 12))
		self.clear_button.grid(row=0, column=1)
		
		self.canvas = tk.Canvas(self.root, bg='white', width=300, height=300)
		self.canvas.grid(row=1, columnspan=2)
		
		self.label = tk.Label(self.root, bg='black', fg='white', bd=4,
			height=1, width=10, justify=tk.CENTER, text='Prediction: ')
		self.label.config(font=('Calibri', 20))
		self.label.grid(row=2, column=0)
		
		self.predict_label = tk.Label(self.root, bg='black', fg='white', bd=4,
			height=1, width=10, justify=tk.CENTER)
		self.predict_label.config(font=('Calibri', 20))
		self.predict_label.grid(row=2, column=1)
		
		self.old_x = None
		self.old_y = None
		self.canvas.bind('<B1-Motion>', self.paint)
		self.canvas.bind('<B3-Motion>', self.erase)
		self.canvas.bind('<ButtonRelease-1>', self.reset)
		self.canvas.bind('<ButtonRelease-3>', self.reset)
		
		self.network = nn.load(network)
		
	def reset(self, event):
		self.old_x = None
		self.old_y = None
	
	def predict(self):
		self.canvas.postscript(file='images/tmp.ps', colormode='gray')
		img = Image.open("images/tmp.ps")
		img.save('images/out.png', 'png')
		
		img = Image.open('images/out.png')
		img = img.resize((28, 28)).convert('L')
		
		data = np.array(img).reshape(784).astype(float)
		data = np.array([(255 - x) / 255 for x in data])
		
		prediction = self.network.predict(data)
		
		self.predict_label.config(text=str(prediction))
	
	def clear(self):
		self.canvas.delete('all')
		self.predict_label.config(text='')
	
	def paint(self, event):
		self.predict_label.config(text='')
		if self.old_x and self.old_y:
			self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, width=10.0, 
				fill='black', capstyle=tk.ROUND, smooth=tk.TRUE, splinesteps=36)
		self.old_x = event.x
		self.old_y = event.y
	
	def erase(self, event):
		self.predict_label.config(text='')
		if self.old_x and self.old_y:
			self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, width=20.0, 
				fill='white', capstyle=tk.ROUND, smooth=tk.TRUE, splinesteps=36)
		self.old_x = event.x
		self.old_y = event.y
	
	def start(self):
		self.root.mainloop()

if __name__ == '__main__':
	recognizer = Recognizer('network.json')
	recognizer.start()

