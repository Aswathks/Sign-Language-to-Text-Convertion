import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from keras.models import model_from_json
from spellchecker import SpellChecker
import operator
from string import ascii_uppercase

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

class Application:
    def __init__(self):
        self.root = tk.Tk()  # Create root window early to avoid BooleanVar error
        self.auto_correct_enabled = tk.BooleanVar(value=True)  # For toggle button

        self.hs = SpellChecker()
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None

        # Load all models
        self.loaded_model = self.load_model("Models/model_new.json", "Models/model_new.h5")
        self.loaded_model_dru = self.load_model("Models/model-bw_dru.json", "Models/model-bw_dru.h5")
        self.loaded_model_tkdi = self.load_model("Models/model-bw_tkdi.json", "Models/model-bw_tkdi.h5")
        self.loaded_model_smn = self.load_model("Models/model-bw_smn.json", "Models/model-bw_smn.h5")

        self.ct = {'blank': 0}
        for i in ascii_uppercase:
            self.ct[i] = 0

        self.str = ""
        self.word = " "
        self.current_symbol = "Empty"

        # Path to the sign language alphabet image
        self.alphabet_image_path = "sign_language_alphabet.png"

        self.setup_ui()
        self.start_tutorial()
        self.video_loop()

    def load_model(self, json_path, weight_path):
        with open(json_path, "r") as f:
            model_json = f.read()
        model = model_from_json(model_json)
        model.load_weights(weight_path)
        return model

    def setup_ui(self):
        self.root.title("Sign Language To Text Conversion")
        self.root.geometry("1000x800")
        self.root.configure(bg="#f5f5f5")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", font=("Arial", 18))
        style.configure("TButton", font=("Arial", 14, "bold"))

        ttk.Label(self.root, text="Sign Language To Text Conversion",
                  font=("Arial", 26, "bold"), background="#f5f5f5").pack(pady=10)

        main_frame = ttk.Frame(self.root)
        main_frame.pack()

        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, padx=20)

        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, padx=20)

        self.panel = ttk.Label(left_frame)
        self.panel.pack()

        self.panel2 = ttk.Label(right_frame)
        self.panel2.pack()

        text_frame = ttk.Frame(right_frame)
        text_frame.pack(pady=10, anchor="w")

        self.panel3 = ttk.Label(text_frame, text="Character: ")
        self.panel3.pack(anchor="w", pady=(10, 0))

        self.panel4 = ttk.Label(text_frame, text="Word: ")
        self.panel4.pack(anchor="w", pady=10)

        self.panel5 = ttk.Label(text_frame, text="Sentence: ")
        self.panel5.pack(anchor="w", pady=10)

        # Auto-Correct Toggle
        toggle_frame = ttk.Frame(right_frame)
        toggle_frame.pack(pady=10, anchor="w")
        ttk.Checkbutton(toggle_frame, text="Enable Auto-Correction", variable=self.auto_correct_enabled).pack()

        ttk.Label(right_frame, text="Suggestions:", foreground="red",
                  font=("Arial", 20, "bold")).pack(pady=(20, 5))

        self.bt1 = ttk.Button(right_frame, text="", command=lambda: self.select_suggestion(0))
        self.bt1.pack(pady=2)
        self.bt2 = ttk.Button(right_frame, text="", command=lambda: self.select_suggestion(1))
        self.bt2.pack(pady=2)
        self.bt3 = ttk.Button(right_frame, text="", command=lambda: self.select_suggestion(2))
        self.bt3.pack(pady=2)

        # Display Sign Language Alphabet Image
        try:
            self.alphabet_image = Image.open(self.alphabet_image_path)
            self.alphabet_image.thumbnail((200, 200))
            self.alphabet_imgtk = ImageTk.PhotoImage(self.alphabet_image)
            self.alphabet_label = ttk.Label(right_frame, image=self.alphabet_imgtk)
            self.alphabet_label.pack(pady=20)
        except Exception as e:
            print(f"Error loading alphabet image: {e}")

    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            cv2image = cv2.flip(frame, 1)
            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])

            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            roi = cv2image[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
            _, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            self.predict(res)

            self.current_image2 = Image.fromarray(res)
            imgtk2 = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk2
            self.panel2.config(image=imgtk2)

            self.panel3.config(text=f"Character: {self.current_symbol}")
            self.panel4.config(text=f"Word: {self.word}")
            self.panel5.config(text=f"Sentence: {self.str}")

            self.update_suggestions()

        self.root.after(5, self.video_loop)

    def predict(self, test_image):
        test_image = cv2.resize(test_image, (128, 128))
        input_data = test_image.reshape(1, 128, 128, 1)

        result = self.loaded_model.predict(input_data)
        result_dru = self.loaded_model_dru.predict(input_data)
        result_tkdi = self.loaded_model_tkdi.predict(input_data)
        result_smn = self.loaded_model_smn.predict(input_data)

        prediction = {'blank': result[0][0]}
        index = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][index]
            index += 1

        prediction_sorted = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        top_symbol = prediction_sorted[0][0]

        if top_symbol in ['D', 'R', 'U']:
            spec = {'D': result_dru[0][0], 'R': result_dru[0][1], 'U': result_dru[0][2]}
            top_symbol = sorted(spec.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        elif top_symbol in ['D', 'I', 'K', 'T']:
            spec = {'D': result_tkdi[0][0], 'I': result_tkdi[0][1], 'K': result_tkdi[0][2], 'T': result_tkdi[0][3]}
            top_symbol = sorted(spec.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        elif top_symbol in ['M', 'N', 'S']:
            spec = {'M': result_smn[0][0], 'N': result_smn[0][1], 'S': result_smn[0][2]}
            top_symbol = sorted(spec.items(), key=operator.itemgetter(1), reverse=True)[0][0]

        self.current_symbol = top_symbol
        self.ct[self.current_symbol] += 1

        if self.ct[self.current_symbol] > 60:
            self.ct['blank'] = 0
            for i in ascii_uppercase:
                self.ct[i] = 0
            if self.current_symbol != 'blank':
                if len(self.str) > 16:
                    self.str = ""
                self.word += self.current_symbol

    def update_suggestions(self):
        word_clean = self.word.strip()

        if self.auto_correct_enabled.get():
            predicts = list(self.hs.candidates(word_clean))
            best_guess = self.hs.correction(word_clean)
            if best_guess in predicts:
                predicts.remove(best_guess)
            predicts.insert(0, best_guess)
        else:
            predicts = [word_clean]

        self.bt1.config(text=predicts[0] if len(predicts) > 0 else "")
        self.bt2.config(text=predicts[1] if len(predicts) > 1 else "")
        self.bt3.config(text=predicts[2] if len(predicts) > 2 else "")

    def select_suggestion(self, index):
        word_clean = self.word.strip()

        if self.auto_correct_enabled.get():
            predicts = list(self.hs.candidates(word_clean))
            best_guess = self.hs.correction(word_clean)
            if best_guess in predicts:
                predicts.remove(best_guess)
            predicts.insert(0, best_guess)
        else:
            predicts = [word_clean]

        if index < len(predicts):
            self.word = ""
            self.str += " " + predicts[index]

    def start_tutorial(self):
        self.tutorial_step = 0
        self.tutorial_texts = [
            "Welcome to the Sign Language Translator!",
            "Place your hand inside the grid box to start.",
            "Make one alphabet sign at a time and hold steady.",
            "Your gesture will be translated into text on the screen.",
            "Tutorial complete! Start signing!"
        ]
        self.show_tutorial_popup()

    def show_tutorial_popup(self):
        if self.tutorial_step >= len(self.tutorial_texts):
            return
        self.tutorial_window = tk.Toplevel(self.root)
        self.tutorial_window.title("Tutorial")
        self.tutorial_window.geometry("400x200")
        self.tutorial_window.grab_set()

        ttk.Label(self.tutorial_window, text=self.tutorial_texts[self.tutorial_step],
                  wraplength=350, font=("Arial", 14)).pack(pady=20)

        ttk.Button(self.tutorial_window, text="Next", command=self.next_tutorial_step).pack()

    def next_tutorial_step(self):
        self.tutorial_window.destroy()
        self.tutorial_step += 1
        self.show_tutorial_popup()

    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()

print("Starting Application...")
Application().root.mainloop()