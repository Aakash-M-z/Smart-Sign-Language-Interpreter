"""
Text-to-Speech engine for regional language output
"""
from typing import Optional
import os
import sys


class TTSEngine:
    """Text-to-Speech engine wrapper"""
    
    def __init__(self, engine: str = "gtts", language: str = "hi", slow: bool = False):
        """
        Initialize TTS engine
        
        Args:
            engine: TTS engine ('gtts' or 'pyttsx3')
            language: Language code (e.g., 'hi' for Hindi, 'ta' for Tamil, 'te' for Telugu)
            slow: Whether to speak slowly (gTTS only)
        """
        self.engine_type = engine
        self.language = language
        self.slow = slow
        self.engine = None
        
        if engine == "gtts":
            try:
                from gtts import gTTS
                import pygame
                self.gtts = gTTS
                try:
                    pygame.mixer.init()
                    self.pygame = pygame
                except Exception as e:
                    print(f"Warning: Could not initialize pygame mixer: {e}")
                    self.pygame = None
            except ImportError:
                print("Warning: gTTS not available. Install with: pip install gtts pygame")
                self.gtts = None
        elif engine == "pyttsx3":
            try:
                import pyttsx3
                self.pyttsx3 = pyttsx3
                self.engine = pyttsx3.init()
                # Set language if supported
                try:
                    voices = self.engine.getProperty('voices')
                    # Try to find a voice matching the language
                    for voice in voices:
                        if language in voice.id.lower() or language in voice.name.lower():
                            self.engine.setProperty('voice', voice.id)
                            break
                except:
                    pass
            except ImportError:
                print("Warning: pyttsx3 not available. Install with: pip install pyttsx3")
                self.pyttsx3 = None
        else:
            raise ValueError(f"Unsupported TTS engine: {engine}")
    
    def speak(self, text: str, save_audio: bool = False, audio_path: str = "temp_audio.mp3"):
        """
        Convert text to speech
        
        Args:
            text: Text to speak
            save_audio: Whether to save audio to file
            audio_path: Path to save audio file
        """
        if not text or text.strip() == "":
            return
        
        if self.engine_type == "gtts":
            self._speak_gtts(text, save_audio, audio_path)
        elif self.engine_type == "pyttsx3":
            self._speak_pyttsx3(text, save_audio, audio_path)
    
    def _speak_gtts(self, text: str, save_audio: bool, audio_path: str):
        """Speak using gTTS"""
        if self.gtts is None:
            print(f"TTS: {text} (gTTS not available)")
            return
        
        try:
            # Create TTS object
            tts = self.gtts(text=text, lang=self.language, slow=self.slow)
            
            if save_audio:
                tts.save(audio_path)
            
            # Play audio
            if self.pygame is not None:
                try:
                    tts.save("temp_tts.mp3")
                    self.pygame.mixer.music.load("temp_tts.mp3")
                    self.pygame.mixer.music.play()
                    # Wait for playback to finish
                    while self.pygame.mixer.music.get_busy():
                        self.pygame.time.Clock().tick(10)
                    # Clean up
                    try:
                        os.remove("temp_tts.mp3")
                    except:
                        pass
                except Exception as e:
                    print(f"Error playing audio: {e}")
                    print(f"TTS: {text}")
            else:
                print(f"TTS: {text}")
        except Exception as e:
            print(f"Error in gTTS: {e}")
            print(f"TTS: {text}")
    
    def _speak_pyttsx3(self, text: str, save_audio: bool, audio_path: str):
        """Speak using pyttsx3"""
        if self.engine is None:
            print(f"TTS: {text} (pyttsx3 not available)")
            return
        
        try:
            self.engine.say(text)
            if save_audio:
                self.engine.save_to_file(text, audio_path)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Error in pyttsx3: {e}")
            print(f"TTS: {text}")
    
    def stop(self):
        """Stop current speech"""
        if self.engine_type == "pyttsx3" and self.engine:
            try:
                self.engine.stop()
            except:
                pass
        elif self.engine_type == "gtts" and self.pygame:
            try:
                self.pygame.mixer.music.stop()
            except:
                pass


# Regional language mappings
REGIONAL_MAPPINGS = {
    "hi": {  # Hindi
        "A": "अ", "B": "ब", "C": "स", "D": "ड", "E": "ए",
        "F": "फ", "G": "ग", "H": "ह", "I": "इ", "J": "ज",
        "K": "क", "L": "ल", "M": "म", "N": "न", "O": "ओ",
        "P": "प", "Q": "क्यू", "R": "र", "S": "स", "T": "ट",
        "U": "उ", "V": "व", "W": "डब्ल्यू", "X": "एक्स", "Y": "वाई", "Z": "जेड"
    },
    "ta": {  # Tamil
        "A": "அ", "B": "ப", "C": "ச", "D": "ட", "E": "எ",
        "F": "ஃப்", "G": "க", "H": "ஹ", "I": "இ", "J": "ஜ",
        "K": "க", "L": "ல", "M": "ம", "N": "ன", "O": "ஓ",
        "P": "ப", "Q": "க்யு", "R": "ர", "S": "ஸ", "T": "ட",
        "U": "உ", "V": "வ", "W": "டபிள்யு", "X": "எக்ஸ்", "Y": "வை", "Z": "ஜெட்"
    },
    "te": {  # Telugu
        "A": "అ", "B": "బ", "C": "స", "D": "డ", "E": "ఎ",
        "F": "ఫ", "G": "గ", "H": "హ", "I": "ఇ", "J": "జ",
        "K": "క", "L": "ల", "M": "మ", "N": "న", "O": "ఓ",
        "P": "ప", "Q": "క్యూ", "R": "ర", "S": "స", "T": "ట",
        "U": "ఉ", "V": "వ", "W": "డబ్ల్యూ", "X": "ఎక్స్", "Y": "వై", "Z": "జెడ్"
    }
}


def get_regional_text(gesture_label: str, language: str = "hi") -> str:
    """
    Convert gesture label to regional language text
    
    Args:
        gesture_label: Gesture label (e.g., "A", "B")
        language: Language code
        
    Returns:
        Regional language text
    """
    mapping = REGIONAL_MAPPINGS.get(language, REGIONAL_MAPPINGS["hi"])
    return mapping.get(gesture_label.upper(), gesture_label)

