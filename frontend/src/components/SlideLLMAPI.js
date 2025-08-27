import * as fabric from 'fabric';

// LLM API Integration for Slide Manipulation
export class SlideLLMAPI {
  constructor(fabricCanvas, slides, setSlides, currentSlideIndex) {
    this.canvas = fabricCanvas;
    this.slides = slides;
    this.setSlides = setSlides;
    this.currentSlideIndex = currentSlideIndex;
  }

  // Process LLM commands via JSON
  async processLLMCommand(command) {
    try {
      const result = await this.executeCommand(command);
      this.updateSlideData();
      return { success: true, result };
    } catch (error) {
      console.error('LLM Command Error:', error);
      return { success: false, error: error.message };
    }
  }

  // Execute individual command
  async executeCommand(command) {
    const { action, ...params } = command;

    switch (action) {
      case 'add_text':
        return this.addText(params);
      case 'add_shape':
        return this.addShape(params);
      case 'add_image':
        return this.addImage(params);
      case 'modify_element':
        return this.modifyElement(params);
      case 'change_background':
        return this.changeBackground(params);
      case 'move_element':
        return this.moveElement(params);
      case 'delete_element':
        return this.deleteElement(params);
      case 'add_chart':
        return this.addChart(params);
      case 'batch_operations':
        return this.batchOperations(params);
      default:
        throw new Error(`Unknown action: ${action}`);
    }
  }

  // Add text element
  addText({ content, position, style = {} }) {
    const text = new fabric.IText(content, {
      left: position.x || 100,
      top: position.y || 100,
      fontFamily: style.fontFamily || 'Arial',
      fontSize: style.fontSize || 20,
      fill: style.color || '#000000',
      fontWeight: style.bold ? 'bold' : 'normal',
      fontStyle: style.italic ? 'italic' : 'normal',
      textDecoration: style.underline ? 'underline' : '',
      textAlign: style.align || 'left'
    });

    this.canvas.add(text);
    return { id: text.id || Date.now(), type: 'text', element: text };
  }

  // Add shape element
  addShape({ type, position, size, style = {} }) {
    let shape;
    const commonProps = {
      left: position.x || 100,
      top: position.y || 100,
      fill: style.fill || '#3b82f6',
      stroke: style.stroke || '#1e40af',
      strokeWidth: style.strokeWidth || 2
    };

    switch (type) {
      case 'rectangle':
        shape = new fabric.Rect({
          ...commonProps,
          width: size.width || 100,
          height: size.height || 100
        });
        break;
      case 'circle':
        shape = new fabric.Circle({
          ...commonProps,
          radius: size.radius || 50
        });
        break;
      case 'triangle':
        shape = new fabric.Triangle({
          ...commonProps,
          width: size.width || 100,
          height: size.height || 100
        });
        break;
      case 'line':
        shape = new fabric.Line([
          position.x || 100,
          position.y || 100,
          (position.x || 100) + (size.width || 100),
          (position.y || 100) + (size.height || 0)
        ], {
          stroke: style.stroke || '#000000',
          strokeWidth: style.strokeWidth || 2
        });
        break;
      default:
        throw new Error(`Unknown shape type: ${type}`);
    }

    this.canvas.add(shape);
    return { id: shape.id || Date.now(), type: 'shape', element: shape };
  }

  // Add image element
  async addImage({ src, position, size = {} }) {
    return new Promise((resolve, reject) => {
      fabric.Image.fromURL(src, (img) => {
        img.set({
          left: position.x || 100,
          top: position.y || 100,
          scaleX: size.width ? size.width / img.width : 1,
          scaleY: size.height ? size.height / img.height : 1
        });

        this.canvas.add(img);
        resolve({ id: img.id || Date.now(), type: 'image', element: img });
      }, { crossOrigin: 'anonymous' });
    });
  }

  // Modify existing element
  modifyElement({ elementId, properties }) {
    const element = this.findElementById(elementId);
    if (!element) {
      throw new Error(`Element with id ${elementId} not found`);
    }

    // Apply properties
    Object.keys(properties).forEach(key => {
      if (key === 'position') {
        element.set({ left: properties.position.x, top: properties.position.y });
      } else if (key === 'size') {
        if (element.type === 'i-text' || element.type === 'text') {
          element.set({ fontSize: properties.size.fontSize });
        } else {
          element.set({ 
            scaleX: properties.size.width / element.width,
            scaleY: properties.size.height / element.height
          });
        }
      } else {
        element.set({ [key]: properties[key] });
      }
    });

    this.canvas.renderAll();
    return { id: elementId, modified: true };
  }

  // Change background
  changeBackground({ type, value }) {
    if (type === 'color') {
      this.canvas.setBackgroundColor(value, () => {
        this.canvas.renderAll();
      });
    } else if (type === 'gradient') {
      const gradient = new fabric.Gradient({
        type: 'linear',
        coords: { x1: 0, y1: 0, x2: this.canvas.width, y2: this.canvas.height },
        colorStops: value.colorStops
      });
      this.canvas.setBackgroundColor(gradient, () => {
        this.canvas.renderAll();
      });
    } else if (type === 'image') {
      fabric.Image.fromURL(value, (img) => {
        this.canvas.setBackgroundImage(img, () => {
          this.canvas.renderAll();
        });
      });
    }

    return { background: { type, value } };
  }

  // Move element
  moveElement({ elementId, position }) {
    const element = this.findElementById(elementId);
    if (!element) {
      throw new Error(`Element with id ${elementId} not found`);
    }

    element.set({
      left: position.x,
      top: position.y
    });

    this.canvas.renderAll();
    return { id: elementId, moved: true, position };
  }

  // Delete element
  deleteElement({ elementId }) {
    const element = this.findElementById(elementId);
    if (!element) {
      throw new Error(`Element with id ${elementId} not found`);
    }

    this.canvas.remove(element);
    return { id: elementId, deleted: true };
  }

  // Add chart (basic implementation)
  addChart({ type, data, position, size = {} }) {
    // This is a simplified chart implementation
    // In production, you'd integrate with a charting library like Chart.js or D3
    const chartGroup = new fabric.Group();
    
    if (type === 'bar') {
      const maxValue = Math.max(...data.values);
      const barWidth = (size.width || 200) / data.values.length;
      const chartHeight = size.height || 150;

      data.values.forEach((value, index) => {
        const barHeight = (value / maxValue) * chartHeight;
        const bar = new fabric.Rect({
          left: index * barWidth,
          top: chartHeight - barHeight,
          width: barWidth - 2,
          height: barHeight,
          fill: data.colors ? data.colors[index] : '#3b82f6'
        });
        chartGroup.addWithUpdate(bar);
      });
    }

    chartGroup.set({
      left: position.x || 100,
      top: position.y || 100
    });

    this.canvas.add(chartGroup);
    return { id: chartGroup.id || Date.now(), type: 'chart', element: chartGroup };
  }

  // Batch operations
  async batchOperations({ operations }) {
    const results = [];
    for (const operation of operations) {
      try {
        const result = await this.executeCommand(operation);
        results.push({ success: true, result });
      } catch (error) {
        results.push({ success: false, error: error.message });
      }
    }
    return { batchResults: results };
  }

  // Helper methods
  findElementById(id) {
    return this.canvas.getObjects().find(obj => obj.id === id || obj.id === String(id));
  }

  updateSlideData() {
    if (this.canvas && this.setSlides) {
      const slideData = this.canvas.toJSON();
      this.setSlides(prev => prev.map((slide, index) => 
        index === this.currentSlideIndex 
          ? { ...slide, data: slideData }
          : slide
      ));
    }
  }

  // Export slide data for LLM
  exportSlideData() {
    const objects = this.canvas.getObjects().map(obj => ({
      id: obj.id || Date.now(),
      type: obj.type,
      properties: {
        left: obj.left,
        top: obj.top,
        width: obj.width,
        height: obj.height,
        ...obj.toObject()
      }
    }));

    return {
      slideId: this.slides[this.currentSlideIndex]?.id,
      background: this.canvas.backgroundColor,
      elements: objects,
      metadata: {
        width: this.canvas.width,
        height: this.canvas.height,
        lastModified: new Date().toISOString()
      }
    };
  }
}

// Tool calling interface for LLM integration
export const createLLMTools = (api) => {
  return [
    {
      name: 'add_text_to_slide',
      description: 'Add text element to the current slide',
      parameters: {
        type: 'object',
        properties: {
          content: { type: 'string', description: 'Text content' },
          position: {
            type: 'object',
            properties: {
              x: { type: 'number' },
              y: { type: 'number' }
            }
          },
          style: {
            type: 'object',
            properties: {
              fontSize: { type: 'number' },
              color: { type: 'string' },
              fontFamily: { type: 'string' },
              bold: { type: 'boolean' },
              italic: { type: 'boolean' }
            }
          }
        },
        required: ['content']
      },
      function: (params) => api.processLLMCommand({ action: 'add_text', ...params })
    },
    {
      name: 'add_shape_to_slide',
      description: 'Add shape element to the current slide',
      parameters: {
        type: 'object',
        properties: {
          type: { type: 'string', enum: ['rectangle', 'circle', 'triangle', 'line'] },
          position: {
            type: 'object',
            properties: {
              x: { type: 'number' },
              y: { type: 'number' }
            }
          },
          size: {
            type: 'object',
            properties: {
              width: { type: 'number' },
              height: { type: 'number' },
              radius: { type: 'number' }
            }
          },
          style: {
            type: 'object',
            properties: {
              fill: { type: 'string' },
              stroke: { type: 'string' },
              strokeWidth: { type: 'number' }
            }
          }
        },
        required: ['type']
      },
      function: (params) => api.processLLMCommand({ action: 'add_shape', ...params })
    },
    {
      name: 'change_slide_background',
      description: 'Change the background of the current slide',
      parameters: {
        type: 'object',
        properties: {
          type: { type: 'string', enum: ['color', 'gradient', 'image'] },
          value: { type: 'string', description: 'Color hex, gradient config, or image URL' }
        },
        required: ['type', 'value']
      },
      function: (params) => api.processLLMCommand({ action: 'change_background', ...params })
    },
    {
      name: 'move_slide_element',
      description: 'Move an element on the slide',
      parameters: {
        type: 'object',
        properties: {
          elementId: { type: 'string', description: 'ID of the element to move' },
          position: {
            type: 'object',
            properties: {
              x: { type: 'number' },
              y: { type: 'number' }
            }
          }
        },
        required: ['elementId', 'position']
      },
      function: (params) => api.processLLMCommand({ action: 'move_element', ...params })
    }
  ];
};