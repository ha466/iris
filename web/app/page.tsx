'use client'

import { motion, useScroll, useTransform } from 'framer-motion'
import { Button } from "@/components/ui/button"
import { Github, ArrowRight, Cpu, Mic, VolumeIcon as VolumeUp } from 'lucide-react'
import { useState } from 'react'

export default function Home() {
  const { scrollYProgress } = useScroll()
  const opacity = useTransform(scrollYProgress, [0, 0.5], [1, 0])
  const scale = useTransform(scrollYProgress, [0, 0.5], [1, 0.8])
  const [isHovered, setIsHovered] = useState(false)

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 to-indigo-900 text-white overflow-hidden">
      <motion.header 
        className="p-6 sticky top-0 z-50 backdrop-blur-md bg-purple-900/30"
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
      >
        <div className="container mx-auto flex justify-between items-center">
          <motion.h1 
            className="text-4xl font-bold"
            whileHover={{ scale: 1.05 }}
          >
            Iris AI
          </motion.h1>
          <nav>
            <ul className="flex space-x-4">
              <li><a href="#features" className="hover:text-purple-300 transition-colors">Features</a></li>
              <li><a href="#about" className="hover:text-purple-300 transition-colors">About</a></li>
              <li><a href="#contact" className="hover:text-purple-300 transition-colors">Contact</a></li>
            </ul>
          </nav>
        </div>
      </motion.header>

      <main className="container mx-auto px-4 py-12">
        <motion.section 
          className="text-center mb-32 relative"
          style={{ opacity, scale }}
        >
          <motion.h2 
            className="text-6xl font-extrabold mb-4"
            initial={{ y: 50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            Meet Iris
          </motion.h2>
          <motion.p 
            className="text-xl mb-8"
            initial={{ y: 50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            Your intelligent companion in the digital world
          </motion.p>
          <motion.div 
            className="w-64 h-64 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 mx-auto mb-8 relative overflow-hidden"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 260, damping: 20, delay: 0.6 }}
          >
            <motion.div 
              className="absolute inset-0 bg-gradient-to-r from-transparent to-purple-700/50"
              animate={{ 
                rotate: 360,
                scale: [1, 1.2, 1],
              }}
              transition={{ 
                duration: 10, 
                repeat: Infinity,
                repeatType: "reverse",
              }}
            />
          </motion.div>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.8 }}
          >
            <Button 
              className="text-lg px-8 py-4 bg-white text-purple-900 hover:bg-purple-100 transition-colors"
              onMouseEnter={() => setIsHovered(true)}
              onMouseLeave={() => setIsHovered(false)}
            >
              Get Started
              <motion.span
                className="ml-2"
                animate={{ x: isHovered ? 5 : 0 }}
              >
                <ArrowRight className="inline" />
              </motion.span>
            </Button>
          </motion.div>
        </motion.section>

        <motion.section 
          id="features"
          className="grid md:grid-cols-3 gap-8 mb-32"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          {[
            { 
              name: 'Llama 3.3', 
              description: 'Advanced language model capable of understanding context, generating human-like text, and assisting with a wide range of tasks. It excels in natural language processing and can engage in meaningful conversations across various domains.',
              icon: <Cpu className="w-12 h-12 mb-4" />
            },
            { 
              name: 'Whisper-large-v3', 
              description: 'State-of-the-art speech recognition model that can accurately transcribe and translate spoken language. It supports multiple languages and can handle diverse accents, background noise, and complex audio environments.',
              icon: <Mic className="w-12 h-12 mb-4" />
            },
            { 
              name: 'Deep Speech TTS', 
              description: 'High-quality text-to-speech synthesis engine that converts written text into natural-sounding speech. It offers a range of voices and can adjust tone, pitch, and speaking rate for a more personalized and engaging audio experience.',
              icon: <VolumeUp className="w-12 h-12 mb-4" />
            }
          ].map((model, index) => (
            <motion.div 
              key={index}
              className="bg-white bg-opacity-10 p-8 rounded-lg backdrop-blur-md"
              whileHover={{ scale: 1.05, boxShadow: "0 0 20px rgba(167, 139, 250, 0.3)" }}
              transition={{ type: "spring", stiffness: 300, damping: 20 }}
            >
              {model.icon}
              <h3 className="text-2xl font-bold mb-4">{model.name}</h3>
              <p className="text-sm leading-relaxed">{model.description}</p>
            </motion.div>
          ))}
        </motion.section>

        <motion.section 
          id="about"
          className="text-center mb-32"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h2 className="text-4xl font-bold mb-8">About Iris AI</h2>
          <p className="max-w-2xl mx-auto mb-8">
            Iris AI is a cutting-edge artificial intelligence platform that combines the power of advanced language processing, speech recognition, and text-to-speech synthesis. Our mission is to create a seamless and intuitive AI experience that enhances human capabilities across various domains.
          </p>
          <Button 
            className="text-lg px-8 py-4"
            onClick={() => window.open('https://github.com/your-username/iris-ai-intro', '_blank')}
          >
            <Github className="mr-2 h-5 w-5" />
            View on GitHub
          </Button>
        </motion.section>

        <motion.section 
          id="contact"
          className="text-center"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h2 className="text-4xl font-bold mb-8">Get in Touch</h2>
          <p className="max-w-2xl mx-auto mb-8">
            Interested in learning more about Iris AI or integrating our technology into your projects? We'd love to hear from you!
          </p>
          <Button className="text-lg px-8 py-4 bg-white text-purple-900 hover:bg-purple-100 transition-colors">
            Contact Us
          </Button>
        </motion.section>
      </main>

      <footer className="text-center p-6 bg-purple-900/30 backdrop-blur-md">
        <p>&copy; 2024 Iris AI. All rights reserved.</p>
      </footer>
    </div>
  )
}

