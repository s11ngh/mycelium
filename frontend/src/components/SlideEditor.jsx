import React, { useState, useEffect, useRef } from 'react';
import * as fabric from 'fabric';
import { v4 as uuidv4 } from 'uuid';
import { useHotkeys } from 'react-hotkeys-hook';
import { SlideLLMAPI, createLLMTools } from './SlideLLMAPI';

const SlideEditor = () => {
  const canvasRef = useRef(null);
  const fabricCanvasRef = useRef(null);
  const llmAPIRef = useRef(null);
  const [slides, setSlides] = useState([]);
  const [currentSlideIndex, setCurrentSlideIndex] = useState(0);
  const [selectedElement, setSelectedElement] = useState(null);
  const [llmCommand, setLlmCommand] = useState('');
  const [llmResponse, setLlmResponse] = useState('');

  // Initialize canvas
  useEffect(() => {
    if (canvasRef.current && !fabricCanvasRef.current) {
      const canvas = new fabric.Canvas(canvasRef.current, {
        width: 800,
        height: 600,
        backgroundColor: '#ffffff'
      });

      fabricCanvasRef.current = canvas;

      // Initialize LLM API
      llmAPIRef.current = new SlideLLMAPI(canvas, slides, setSlides, currentSlideIndex);

      // Add event listeners
      canvas.on('selection:created', (e) => {
        setSelectedElement(e.selected[0]);
      });

      canvas.on('selection:updated', (e) => {
        setSelectedElement(e.selected[0]);
      });

      canvas.on('selection:cleared', () => {
        setSelectedElement(null);
      });

      // Create initial slide if none exist
      if (slides.length === 0) {
        createNewSlide();
      }

      return () => {
        canvas.dispose();
      };
    }
  }, []);

  // Update LLM API when slides or current slide changes
  useEffect(() => {
    if (llmAPIRef.current) {
      llmAPIRef.current.slides = slides;
      llmAPIRef.current.currentSlideIndex = currentSlideIndex;
    }
  }, [slides, currentSlideIndex]);

  // Load slide data when current slide changes
  useEffect(() => {
    if (fabricCanvasRef.current && slides[currentSlideIndex]) {
      const canvas = fabricCanvasRef.current;
      canvas.loadFromJSON(slides[currentSlideIndex].data, () => {
        canvas.renderAll();
      });
    }
  }, [currentSlideIndex, slides]);

  // Save current slide data
  const saveCurrentSlide = () => {
    if (fabricCanvasRef.current && slides[currentSlideIndex]) {
      const canvas = fabricCanvasRef.current;
      const slideData = canvas.toJSON();
      
      setSlides(prev => prev.map((slide, index) => 
        index === currentSlideIndex 
          ? { ...slide, data: slideData, thumbnail: canvas.toDataURL({ multiplier: 0.2 }) }
          : slide
      ));
    }
  };

  // Create new slide
  const createNewSlide = () => {
    const newSlide = {
      id: uuidv4(),
      title: `Slide ${slides.length + 1}`,
      data: {
        version: '5.2.4',
        objects: [],
        background: '#ffffff'
      },
      thumbnail: null
    };

    setSlides(prev => [...prev, newSlide]);
    
    if (slides.length === 0) {
      setCurrentSlideIndex(0);
    }
  };

  // Delete slide
  const deleteSlide = (index) => {
    if (slides.length > 1) {
      setSlides(prev => prev.filter((_, i) => i !== index));
      if (currentSlideIndex >= index && currentSlideIndex > 0) {
        setCurrentSlideIndex(prev => prev - 1);
      }
    }
  };

  // Navigate to slide
  const goToSlide = (index) => {
    saveCurrentSlide();
    setCurrentSlideIndex(index);
  };

  // Add text element
  const addText = () => {
    if (fabricCanvasRef.current) {
      const text = new fabric.IText('Click to edit text', {
        left: 100,
        top: 100,
        fontFamily: 'Arial',
        fontSize: 20,
        fill: '#000000'
      });

      fabricCanvasRef.current.add(text);
      fabricCanvasRef.current.setActiveObject(text);
    }
  };

  // Add rectangle
  const addRectangle = () => {
    if (fabricCanvasRef.current) {
      const rect = new fabric.Rect({
        left: 100,
        top: 100,
        width: 100,
        height: 100,
        fill: '#3b82f6',
        stroke: '#1e40af',
        strokeWidth: 2
      });

      fabricCanvasRef.current.add(rect);
      fabricCanvasRef.current.setActiveObject(rect);
    }
  };

  // Add circle
  const addCircle = () => {
    if (fabricCanvasRef.current) {
      const circle = new fabric.Circle({
        left: 100,
        top: 100,
        radius: 50,
        fill: '#ef4444',
        stroke: '#dc2626',
        strokeWidth: 2
      });

      fabricCanvasRef.current.add(circle);
      fabricCanvasRef.current.setActiveObject(circle);
    }
  };

  // Delete selected element
  const deleteSelected = () => {
    if (fabricCanvasRef.current && selectedElement) {
      fabricCanvasRef.current.remove(selectedElement);
      setSelectedElement(null);
    }
  };

  // Execute LLM command
  const executeLLMCommand = async () => {
    if (!llmCommand.trim() || !llmAPIRef.current) return;

    try {
      setLlmResponse('Processing command...');
      
      // Parse JSON command
      let command;
      try {
        command = JSON.parse(llmCommand);
      } catch (error) {
        // If not JSON, try to interpret as natural language and convert to command
        command = parseNaturalLanguageCommand(llmCommand);
      }

      const result = await llmAPIRef.current.processLLMCommand(command);
      
      if (result.success) {
        setLlmResponse(`✅ Command executed successfully: ${JSON.stringify(result.result, null, 2)}`);
        saveCurrentSlide();
      } else {
        setLlmResponse(`❌ Error: ${result.error}`);
      }
    } catch (error) {
      setLlmResponse(`❌ Error: ${error.message}`);
    }
  };

  // Parse natural language commands to JSON
  const parseNaturalLanguageCommand = (command) => {
    const lowerCommand = command.toLowerCase();
    
    if (lowerCommand.includes('add text')) {
      const textMatch = command.match(/"([^"]+)"/);
      const text = textMatch ? textMatch[1] : 'Sample text';
      return {
        action: 'add_text',
        content: text,
        position: { x: 100, y: 100 }
      };
    } else if (lowerCommand.includes('add rectangle') || lowerCommand.includes('add rect')) {
      return {
        action: 'add_shape',
        type: 'rectangle',
        position: { x: 100, y: 100 },
        size: { width: 100, height: 100 }
      };
    } else if (lowerCommand.includes('add circle')) {
      return {
        action: 'add_shape',
        type: 'circle',
        position: { x: 100, y: 100 },
        size: { radius: 50 }
      };
    } else if (lowerCommand.includes('change background')) {
      const colorMatch = command.match(/#[0-9a-fA-F]{6}/);
      const color = colorMatch ? colorMatch[0] : '#f0f0f0';
      return {
        action: 'change_background',
        type: 'color',
        value: color
      };
    }
    
    throw new Error('Could not parse command. Please use JSON format or supported natural language commands.');
  };

  // Change background color
  const changeBackground = (color) => {
    if (fabricCanvasRef.current) {
      fabricCanvasRef.current.setBackgroundColor(color, () => {
        fabricCanvasRef.current.renderAll();
      });
    }
  };

  // Get sample commands for LLM
  const getSampleCommands = () => [
    {
      name: 'Add Text',
      command: JSON.stringify({
        action: 'add_text',
        content: 'Hello World',
        position: { x: 150, y: 100 },
        style: { fontSize: 24, color: '#333333' }
      }, null, 2)
    },
    {
      name: 'Add Rectangle',
      command: JSON.stringify({
        action: 'add_shape',
        type: 'rectangle',
        position: { x: 200, y: 200 },
        size: { width: 150, height: 100 },
        style: { fill: '#3b82f6', stroke: '#1e40af' }
      }, null, 2)
    },
    {
      name: 'Change Background',
      command: JSON.stringify({
        action: 'change_background',
        type: 'color',
        value: '#f8fafc'
      }, null, 2)
    }
  ];

  // Keyboard shortcuts
  useHotkeys('ctrl+n', (e) => {
    e.preventDefault();
    createNewSlide();
  });

  useHotkeys('delete', (e) => {
    e.preventDefault();
    deleteSelected();
  });

  useHotkeys('ctrl+s', (e) => {
    e.preventDefault();
    saveCurrentSlide();
  });

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Slide Sidebar */}
      <div className="w-64 bg-white border-r border-gray-300 overflow-y-auto">
        <div className="p-4">
          <button
            onClick={createNewSlide}
            className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors mb-4"
          >
            + New Slide
          </button>
          
          <div className="space-y-2">
            {slides.map((slide, index) => (
              <div
                key={slide.id}
                className={`relative border-2 rounded-lg cursor-pointer transition-colors ${
                  index === currentSlideIndex 
                    ? 'border-blue-500 bg-blue-50' 
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => goToSlide(index)}
              >
                <div className="p-2">
                  <div className="text-sm font-medium">{slide.title}</div>
                  <div className="w-full h-16 bg-gray-100 rounded mt-1 flex items-center justify-center">
                    {slide.thumbnail ? (
                      <img src={slide.thumbnail} alt={slide.title} className="max-w-full max-h-full" />
                    ) : (
                      <span className="text-gray-400">Preview</span>
                    )}
                  </div>
                </div>
                {slides.length > 1 && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteSlide(index);
                    }}
                    className="absolute top-1 right-1 bg-red-500 text-white w-5 h-5 rounded-full text-xs hover:bg-red-600"
                  >
                    ×
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Editor */}
      <div className="flex-1 flex flex-col">
        {/* Toolbar */}
        <div className="bg-white border-b border-gray-300 p-4">
          <div className="flex items-center space-x-2">
            <button
              onClick={addText}
              className="bg-gray-600 text-white px-3 py-1 rounded hover:bg-gray-700 transition-colors"
            >
              Add Text
            </button>
            <button
              onClick={addRectangle}
              className="bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700 transition-colors"
            >
              Rectangle
            </button>
            <button
              onClick={addCircle}
              className="bg-red-600 text-white px-3 py-1 rounded hover:bg-red-700 transition-colors"
            >
              Circle
            </button>
            
            <div className="border-l border-gray-300 pl-2 ml-2">
              <label className="text-sm text-gray-600 mr-2">Background:</label>
              <input
                type="color"
                defaultValue="#ffffff"
                onChange={(e) => changeBackground(e.target.value)}
                className="w-8 h-8 rounded cursor-pointer"
              />
            </div>

            {selectedElement && (
              <button
                onClick={deleteSelected}
                className="bg-red-600 text-white px-3 py-1 rounded hover:bg-red-700 transition-colors ml-auto"
              >
                Delete Selected
              </button>
            )}
          </div>
        </div>

        {/* Canvas Area */}
        <div className="flex-1 flex">
          <div className="flex-1 flex items-center justify-center p-8">
            <div className="bg-white shadow-lg">
              <canvas ref={canvasRef} />
            </div>
          </div>

          {/* LLM Command Panel */}
          <div className="w-80 bg-white border-l border-gray-300 p-4 overflow-y-auto">
            <h3 className="text-lg font-semibold mb-4">LLM Command Interface</h3>
            
            {/* Sample Commands */}
            <div className="mb-4">
              <label className="text-sm font-medium text-gray-700 mb-2 block">Sample Commands:</label>
              <div className="space-y-2">
                {getSampleCommands().map((sample, index) => (
                  <button
                    key={index}
                    onClick={() => setLlmCommand(sample.command)}
                    className="w-full text-left bg-gray-100 hover:bg-gray-200 p-2 rounded text-xs"
                  >
                    {sample.name}
                  </button>
                ))}
              </div>
            </div>

            {/* Command Input */}
            <div className="mb-4">
              <label className="text-sm font-medium text-gray-700 mb-2 block">JSON Command or Natural Language:</label>
              <textarea
                value={llmCommand}
                onChange={(e) => setLlmCommand(e.target.value)}
                placeholder='{"action": "add_text", "content": "Hello"} or "add text Hello"'
                className="w-full h-32 p-2 border border-gray-300 rounded text-sm font-mono"
              />
              <button
                onClick={executeLLMCommand}
                className="w-full mt-2 bg-green-600 text-white py-2 px-4 rounded hover:bg-green-700 transition-colors"
              >
                Execute Command
              </button>
            </div>

            {/* Response Area */}
            {llmResponse && (
              <div className="mb-4">
                <label className="text-sm font-medium text-gray-700 mb-2 block">Response:</label>
                <div className="bg-gray-100 p-2 rounded text-sm font-mono whitespace-pre-wrap max-h-32 overflow-y-auto">
                  {llmResponse}
                </div>
              </div>
            )}

            {/* API Documentation */}
            <div className="text-xs text-gray-600">
              <h4 className="font-medium mb-2">Available Actions:</h4>
              <ul className="space-y-1">
                <li>• add_text: Add text with content, position, style</li>
                <li>• add_shape: Add rectangle, circle, triangle, line</li>
                <li>• add_image: Add image with src and position</li>
                <li>• change_background: Change slide background</li>
                <li>• move_element: Move existing element</li>
                <li>• modify_element: Modify element properties</li>
                <li>• delete_element: Delete specific element</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Status Bar */}
        <div className="bg-gray-50 border-t border-gray-300 p-2 text-sm text-gray-600">
          Slide {currentSlideIndex + 1} of {slides.length} | 
          Press Ctrl+N for new slide, Delete to remove selected element, Ctrl+S to save
        </div>
      </div>
    </div>
  );
};

export default SlideEditor;