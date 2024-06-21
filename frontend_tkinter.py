import tkinter as tk


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drawing App")

        self.canvas = tk.Canvas(root, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.pen_color = "black"
        self.drawing = False
        self.last_x, self.last_y = None, None

        self.lines = []  # List to store drawn lines

        self.undo_button = tk.Button(root, text="Undo", command=self.undo)
        self.erase_button = tk.Button(root, text="Erase", command=self.erase)

        self.undo_button.pack(side=tk.LEFT)
        self.erase_button.pack(side=tk.LEFT)

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

    def start_draw(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            line = self.canvas.create_line(
                self.last_x, self.last_y, x, y, fill=self.pen_color, width=2
            )
            self.lines.append(line)  # Store the drawn line
            self.last_x, self.last_y = x, y

    def stop_draw(self, event):
        self.drawing = False

    def undo(self):
        if self.lines:
            last_line = self.lines.pop()  # Remove the last drawn line
            self.canvas.delete(last_line)

    def erase(self):
        self.canvas.delete("all")


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
