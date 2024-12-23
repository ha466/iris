'use client'

import { motion, useScroll, useTransform, AnimatePresence } from 'framer-motion'
import { Button } from "@/components/ui/button"
import { Github, ArrowRight, Cpu, Mic, VolumeIcon as VolumeUp, Brain, Sparkles } from 'lucide-react'
import { useState, useEffect } from 'react'

export default function Home() {
  const { scrollYProgress } = useScroll()
  const opacity = useTransform(scrollYProgress, [0, 0.5], [1, 0])
  const scale = useTransform(scrollYProgress, [0, 0.5], [1, 0.8])
  const [isHovered, setIsHovered] = useState(false)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        delayChildren: 0.3,
        staggerChildren: 0.2
      }
    }
  }

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1
    }
  }

  if (!mounted) return null

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-indigo-900 to-blue-900 text-white overflow-hidden">
      <motion.header 
        className="p-6 sticky top-0 z-50 backdrop-blur-md bg-purple-900/30"
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
      >
        <div className="container mx-auto flex justify-between items-center">
          <motion.div 
            className="flex items-center gap-2"
            whileHover={{ scale: 1.05 }}
          >
            <Brain className="w-8 h-8" />
            <h1 className="text-4xl font-bold">Iris AI</h1>
          </motion.div>
          <nav>
            <ul className="flex space-x-6">
              <motion.li whileHover={{ scale: 1.1 }}>
                <a href="#features" className="hover:text-purple-300 transition-colors">Features</a>
              </motion.li>
              <motion.li whileHover={{ scale: 1.1 }}>
                <a href="#about" className="hover:text-purple-300 transition-colors">About</a>
              </motion.li>
              <motion.li whileHover={{ scale: 1.1 }}>
                <a href="#contact" className="hover:text-purple-300 transition-colors">Contact</a>
              </motion.li>
            </ul>
          </nav>
        </div>
      </motion.header>

      <main className="container mx-auto px-4 py-12">
        <motion.section 
          className="text-center mb-32 relative"
          style={{ opacity, scale }}
        >
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="space-y-6"
          >
            <motion.h2 
              variants={itemVariants}
              className="text-7xl font-extrabold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-600"
            >
              Meet Iris
            </motion.h2>
            <motion.p 
              variants={itemVariants}
              className="text-2xl mb-8 text-gray-300"
            >
              Your intelligent companion in the digital world
            </motion.p>
            <motion.div 
              variants={itemVariants}
              className="relative w-80 h-80 mx-auto mb-12"
            >
              <motion.div 
                className="absolute inset-0 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 opacity-75"
                animate={{ 
                  scale: [1, 1.2, 1],
                  rotate: 360,
                }}
                transition={{ 
                  duration: 20,
                  repeat: Infinity,
                  repeatType: "reverse",
                }}
              />
              <motion.div 
                className="absolute inset-0 rounded-full overflow-hidden"
                whileHover={{ scale: 1.05 }}
              >
                <img 
                  src="/iris-ai.png" 
                  alt="Iris AI Visualization"
                  className="w-full h-full object-cover"
                />
              </motion.div>
              <motion.div
                className="absolute -inset-4"
                animate={{
                  rotate: [0, 360],
                }}
                transition={{
                  duration: 30,
                  repeat: Infinity,
                  ease: "linear",
                }}
              >
                {[...Array(8)].map((_, i) => (
                  <motion.div
                    key={i}
                    className="absolute top-1/2 left-1/2 w-2 h-2"
                    style={{
                      rotate: `${i * 45}deg`,
                      translateX: '-50%',
                      translateY: '-50%',
                    }}
                  >
                    <motion.div
                      className="w-2 h-2 bg-purple-400 rounded-full"
                      animate={{
                        scale: [1, 1.5, 1],
                        opacity: [0.5, 1, 0.5],
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        delay: i * 0.2,
                      }}
                    />
                  </motion.div>
                ))}
              </motion.div>
            </motion.div>
            <motion.div
              variants={itemVariants}
              className="flex justify-center gap-4"
            >
              <Button 
                className="text-lg px-8 py-6 bg-white text-purple-900 hover:bg-purple-100 transition-colors"
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
              <Button 
                variant="outline"
                className="text-lg px-8 py-6 border-2"
                onClick={() => window.open('https://github.com/ha466/iris.git', '_blank')}
              >
                <Github className="mr-2 h-5 w-5" />
                View on GitHub
              </Button>
            </motion.div>
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
              description: 'Advanced language model capable of understanding context, generating human-like text, and assisting with a wide range of tasks.',
              icon: <Cpu className="w-12 h-12 mb-4" />,
              gradient: 'from-purple-500 to-pink-500'
            },
            { 
              name: 'Whisper-large-v3', 
              description: 'State-of-the-art speech recognition model that can accurately transcribe and translate spoken language.',
              icon: <Mic className="w-12 h-12 mb-4" />,
              gradient: 'from-blue-500 to-purple-500'
            },
            { 
              name: 'Deep Speech TTS', 
              description: 'High-quality text-to-speech synthesis engine that converts written text into natural-sounding speech.',
              icon: <VolumeUp className="w-12 h-12 mb-4" />,
              gradient: 'from-pink-500 to-orange-500'
            }
          ].map((model, index) => (
            <motion.div 
              key={index}
              className={`relative p-8 rounded-xl overflow-hidden`}
              whileHover={{ scale: 1.05 }}
              transition={{ type: "spring", stiffness: 300, damping: 20 }}
            >
              <motion.div 
                className={`absolute inset-0 bg-gradient-to-br ${model.gradient} opacity-10`}
                initial={{ opacity: 0 }}
                animate={{ opacity: 0.1 }}
                whileHover={{ opacity: 0.2 }}
              />
              <div className="relative z-10">
                {model.icon}
                <h3 className="text-2xl font-bold mb-4">{model.name}</h3>
                <p className="text-gray-300 leading-relaxed">{model.description}</p>
              </div>
            </motion.div>
          ))}
        </motion.section>

        <motion.section 
          id="about"
          className="text-center mb-32 relative"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <motion.div
            className="absolute top-0 right-0 -translate-y-1/2"
            animate={{
              scale: [1, 1.2, 1],
              rotate: 360,
            }}
            transition={{
              duration: 20,
              repeat: Infinity,
              repeatType: "reverse",
            }}
          >
            <Sparkles className="w-24 h-24 text-purple-500 opacity-20" />
          </motion.div>
          <h2 className="text-5xl font-bold mb-8">About Iris AI</h2>
          <p className="max-w-2xl mx-auto mb-8 text-gray-300 text-lg leading-relaxed">
            Iris AI is a cutting-edge artificial intelligence platform that combines the power of advanced language processing, 
            speech recognition, and text-to-speech synthesis. Our mission is to create a seamless and intuitive AI experience 
            that enhances human capabilities across various domains.
          </p>
          <Button 
            className="text-lg px-8 py-6"
            onClick={() => window.open('https://github.com/ha466/iris.git', '_blank')}
          >
            <Github className="mr-2 h-5 w-5" />
            View on GitHub
          </Button>
        </motion.section>

        <motion.section 
          id="contact"
          className="text-center relative"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <div className="max-w-4xl mx-auto bg-white/5 backdrop-blur-lg rounded-2xl p-12 border border-white/10">
            <h2 className="text-4xl font-bold mb-8">Get in Touch</h2>
            <p className="max-w-2xl mx-auto mb-8 text-gray-300">
              Interested in learning more about Iris AI or integrating our technology into your projects? We'd love to hear from you!
            </p>
            <Button className="text-lg px-8 py-6 bg-white text-purple-900 hover:bg-purple-100 transition-colors">
              Contact Us
            </Button>
          </div>
        </motion.section>
      </main>

      <footer className="mt-32 border-t border-white/10">
        <div className="container mx-auto px-4 py-8">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-2">
              <Brain className="w-6 h-6" />
              <span className="font-bold">Iris AI</span>
            </div>
            <p className="text-gray-400">&copy; 2024 Iris AI. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}

