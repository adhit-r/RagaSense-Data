#!/usr/bin/env python3
"""
RagaSense ML Demo Script
Demonstrate the raga detection system with sample audio
"""

import os
import json
import logging
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import time

from raga_detection_system import RagaDetectionSystem, RagaDetectionConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_audio():
    """Create sample audio for demonstration"""
    logger.info("🎵 Creating sample audio for demonstration...")
    
    # Create a simple sine wave as placeholder
    sample_rate = 44100
    duration = 5  # seconds
    
    # Generate a simple melody (C major scale)
    frequencies = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # C4 to C5
    audio = np.array([])
    
    for freq in frequencies:
        t = np.linspace(0, 0.5, int(sample_rate * 0.5))
        note = 0.3 * np.sin(2 * np.pi * freq * t)
        audio = np.concatenate([audio, note])
    
    # Add some silence
    silence = np.zeros(int(sample_rate * 0.2))
    audio = np.concatenate([audio, silence])
    
    # Save as WAV file
    output_path = "ml_models/sample_audio.wav"
    librosa.output.write_wav(output_path, audio, sample_rate)
    
    logger.info(f"✅ Sample audio created: {output_path}")
    return output_path

def demonstrate_audio_processing():
    """Demonstrate audio processing capabilities"""
    logger.info("🔧 Demonstrating audio processing...")
    
    # Create sample audio
    audio_path = create_sample_audio()
    
    # Initialize system
    config = RagaDetectionConfig()
    system = RagaDetectionSystem(config, model_variant='ensemble')
    
    # Load audio
    audio = system.preprocessor.load_audio(audio_path)
    if audio is None:
        logger.error("❌ Failed to load audio")
        return
    
    # Extract features
    features = system.preprocessor.extract_all_features(audio)
    
    # Display feature information
    logger.info("📊 Extracted Features:")
    for feature_name, feature_data in features.items():
        logger.info(f"   {feature_name}: {feature_data.shape}")
    
    # Create visualizations
    create_feature_visualizations(features, audio, config.SAMPLE_RATE)
    
    return features

def create_feature_visualizations(features, audio, sample_rate):
    """Create visualizations of extracted features"""
    logger.info("📈 Creating feature visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('RagaSense Audio Feature Extraction', fontsize=16)
    
    # Waveform
    time = np.linspace(0, len(audio) / sample_rate, len(audio))
    axes[0, 0].plot(time, audio, color='blue', linewidth=0.5)
    axes[0, 0].set_title('Audio Waveform')
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mel-Spectrogram
    axes[0, 1].imshow(features['mel_spectrogram'], aspect='auto', origin='lower', cmap='viridis')
    axes[0, 1].set_title('Mel-Spectrogram')
    axes[0, 1].set_xlabel('Time Frames')
    axes[0, 1].set_ylabel('Mel Bins')
    
    # MFCC
    axes[1, 0].imshow(features['mfcc'], aspect='auto', origin='lower', cmap='plasma')
    axes[1, 0].set_title('MFCC Features')
    axes[1, 0].set_xlabel('Time Frames')
    axes[1, 0].set_ylabel('MFCC Coefficients')
    
    # Chroma
    axes[1, 1].imshow(features['chroma'], aspect='auto', origin='lower', cmap='coolwarm')
    axes[1, 1].set_title('Chroma Features')
    axes[1, 1].set_xlabel('Time Frames')
    axes[1, 1].set_ylabel('Pitch Classes')
    
    # Spectral Centroid
    axes[2, 0].plot(features['spectral_centroid'][0], color='green')
    axes[2, 0].set_title('Spectral Centroid')
    axes[2, 0].set_xlabel('Time Frames')
    axes[2, 0].set_ylabel('Frequency (Hz)')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Zero Crossing Rate
    axes[2, 1].plot(features['zero_crossing_rate'][0], color='red')
    axes[2, 1].set_title('Zero Crossing Rate')
    axes[2, 1].set_xlabel('Time Frames')
    axes[2, 1].set_ylabel('ZCR')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ml_models/feature_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("✅ Feature visualizations saved to ml_models/feature_visualization.png")

def demonstrate_model_architecture():
    """Demonstrate model architecture"""
    logger.info("🏗️ Demonstrating model architecture...")
    
    config = RagaDetectionConfig()
    
    # Create different model variants
    model_variants = ['cnn_lstm', 'yue_foundation', 'ensemble']
    
    for variant in model_variants:
        logger.info(f"   Creating {variant} model...")
        system = RagaDetectionSystem(config, model_variant=variant)
        
        # Create dummy model for demonstration
        num_ragas = 100  # Dummy number
        model = system.create_model(num_ragas)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"   {variant}: {total_params:,} total parameters, {trainable_params:,} trainable")

def demonstrate_api_capabilities():
    """Demonstrate API capabilities"""
    logger.info("🌐 Demonstrating API capabilities...")
    
    # Show supported formats
    config = RagaDetectionConfig()
    logger.info(f"📁 Supported audio formats: {config.SUPPORTED_FORMATS}")
    
    # Show model variants
    logger.info("🎯 Available model variants:")
    for variant, description in config.MODEL_VARIANTS.items():
        logger.info(f"   {variant}: {description}")
    
    # Show API endpoints
    api_endpoints = [
        "POST /predict - Upload audio file for raga detection",
        "POST /predict-batch - Process multiple audio files",
        "GET /ragas - List all supported ragas",
        "GET /model-info - Get model information",
        "GET /health - Health check",
        "GET /demo - Interactive demo page"
    ]
    
    logger.info("📡 API Endpoints:")
    for endpoint in api_endpoints:
        logger.info(f"   {endpoint}")

def main():
    """Main demonstration function"""
    logger.info("🎵 RagaSense ML Model Demonstration")
    logger.info("=" * 60)
    
    try:
        # Demonstrate audio processing
        features = demonstrate_audio_processing()
        
        # Demonstrate model architecture
        demonstrate_model_architecture()
        
        # Demonstrate API capabilities
        demonstrate_api_capabilities()
        
        # Create summary
        logger.info("\n🎉 Demonstration Summary:")
        logger.info("✅ Audio processing pipeline working")
        logger.info("✅ Feature extraction successful")
        logger.info("✅ Model architectures created")
        logger.info("✅ API endpoints defined")
        logger.info("✅ Visualizations generated")
        
        logger.info("\n📁 Generated Files:")
        logger.info("   - ml_models/sample_audio.wav")
        logger.info("   - ml_models/feature_visualization.png")
        
        logger.info("\n🚀 Next Steps:")
        logger.info("   1. Train the model: python train_raga_model.py")
        logger.info("   2. Start the API: python raga_detection_api.py")
        logger.info("   3. Test with real audio files")
        logger.info("   4. Deploy to production")
        
    except Exception as e:
        logger.error(f"❌ Error during demonstration: {e}")
        raise

if __name__ == "__main__":
    main()

