const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { v4: uuidv4 } = require('uuid');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ extended: true }));

// In-memory storage (In production, use a proper database)
let presentations = {};
let templates = {};

// Initialize with sample templates
templates['default'] = {
  id: 'default',
  name: 'Default Template',
  slides: [
    {
      id: uuidv4(),
      title: 'Title Slide',
      data: {
        version: '5.2.4',
        objects: [
          {
            type: 'IText',
            left: 200,
            top: 250,
            width: 400,
            height: 50,
            text: 'Presentation Title',
            fontSize: 36,
            fontFamily: 'Arial',
            fill: '#333333',
            textAlign: 'center'
          }
        ],
        background: '#ffffff'
      }
    }
  ]
};

// Routes

// Get all presentations
app.get('/api/presentations', (req, res) => {
  res.json(Object.values(presentations));
});

// Get specific presentation
app.get('/api/presentations/:id', (req, res) => {
  const presentation = presentations[req.params.id];
  if (!presentation) {
    return res.status(404).json({ error: 'Presentation not found' });
  }
  res.json(presentation);
});

// Create new presentation
app.post('/api/presentations', (req, res) => {
  const { title, templateId } = req.body;
  const id = uuidv4();
  
  let slides = [];
  if (templateId && templates[templateId]) {
    slides = JSON.parse(JSON.stringify(templates[templateId].slides));
  } else {
    // Default empty slide
    slides = [{
      id: uuidv4(),
      title: 'Slide 1',
      data: {
        version: '5.2.4',
        objects: [],
        background: '#ffffff'
      },
      thumbnail: null
    }];
  }

  const presentation = {
    id,
    title: title || 'Untitled Presentation',
    slides,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  };

  presentations[id] = presentation;
  res.status(201).json(presentation);
});

// Update presentation
app.put('/api/presentations/:id', (req, res) => {
  const presentation = presentations[req.params.id];
  if (!presentation) {
    return res.status(404).json({ error: 'Presentation not found' });
  }

  const { title, slides } = req.body;
  presentation.title = title || presentation.title;
  presentation.slides = slides || presentation.slides;
  presentation.updatedAt = new Date().toISOString();

  res.json(presentation);
});

// Delete presentation
app.delete('/api/presentations/:id', (req, res) => {
  if (!presentations[req.params.id]) {
    return res.status(404).json({ error: 'Presentation not found' });
  }

  delete presentations[req.params.id];
  res.status(204).send();
});

// LLM Integration Endpoints

// Process LLM command for a specific slide
app.post('/api/presentations/:id/slides/:slideId/llm-command', (req, res) => {
  const presentation = presentations[req.params.id];
  if (!presentation) {
    return res.status(404).json({ error: 'Presentation not found' });
  }

  const slideIndex = presentation.slides.findIndex(s => s.id === req.params.slideId);
  if (slideIndex === -1) {
    return res.status(404).json({ error: 'Slide not found' });
  }

  const { command } = req.body;
  
  try {
    const result = processLLMCommand(command, presentation.slides[slideIndex]);
    presentation.updatedAt = new Date().toISOString();
    
    res.json({
      success: true,
      result,
      slide: presentation.slides[slideIndex]
    });
  } catch (error) {
    res.status(400).json({
      success: false,
      error: error.message
    });
  }
});

// Batch LLM commands
app.post('/api/presentations/:id/llm-batch', (req, res) => {
  const presentation = presentations[req.params.id];
  if (!presentation) {
    return res.status(404).json({ error: 'Presentation not found' });
  }

  const { commands } = req.body;
  const results = [];

  try {
    commands.forEach((cmd, index) => {
      const { slideId, command } = cmd;
      const slideIndex = presentation.slides.findIndex(s => s.id === slideId);
      
      if (slideIndex !== -1) {
        try {
          const result = processLLMCommand(command, presentation.slides[slideIndex]);
          results.push({ success: true, result, slideIndex: index });
        } catch (error) {
          results.push({ success: false, error: error.message, slideIndex: index });
        }
      } else {
        results.push({ success: false, error: 'Slide not found', slideIndex: index });
      }
    });

    presentation.updatedAt = new Date().toISOString();
    res.json({ results, presentation });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Get templates
app.get('/api/templates', (req, res) => {
  res.json(Object.values(templates));
});

// Export presentation (placeholder for future implementation)
app.post('/api/presentations/:id/export', (req, res) => {
  const { format } = req.body; // 'pdf', 'pptx', 'images'
  
  // This would implement actual export functionality
  res.json({
    message: `Export to ${format} functionality would be implemented here`,
    downloadUrl: `#export-${req.params.id}-${format}`
  });
});

// Helper function to process LLM commands
function processLLMCommand(command, slide) {
  const { action, ...params } = command;

  switch (action) {
    case 'add_text':
      return addTextToSlide(params, slide);
    case 'add_shape':
      return addShapeToSlide(params, slide);
    case 'change_background':
      return changeSlideBackground(params, slide);
    case 'add_image':
      return addImageToSlide(params, slide);
    case 'modify_element':
      return modifySlideElement(params, slide);
    case 'delete_element':
      return deleteSlideElement(params, slide);
    default:
      throw new Error(`Unknown action: ${action}`);
  }
}

function addTextToSlide({ content, position, style }, slide) {
  const element = {
    type: 'IText',
    left: position?.x || 100,
    top: position?.y || 100,
    text: content,
    fontSize: style?.fontSize || 20,
    fontFamily: style?.fontFamily || 'Arial',
    fill: style?.color || '#000000',
    fontWeight: style?.bold ? 'bold' : 'normal',
    fontStyle: style?.italic ? 'italic' : 'normal',
    textAlign: style?.align || 'left',
    id: Date.now()
  };

  slide.data.objects.push(element);
  return { id: element.id, type: 'text', element };
}

function addShapeToSlide({ type, position, size, style }, slide) {
  let element = {
    left: position?.x || 100,
    top: position?.y || 100,
    fill: style?.fill || '#3b82f6',
    stroke: style?.stroke || '#1e40af',
    strokeWidth: style?.strokeWidth || 2,
    id: Date.now()
  };

  switch (type) {
    case 'rectangle':
      element.type = 'Rect';
      element.width = size?.width || 100;
      element.height = size?.height || 100;
      break;
    case 'circle':
      element.type = 'Circle';
      element.radius = size?.radius || 50;
      break;
    case 'triangle':
      element.type = 'Triangle';
      element.width = size?.width || 100;
      element.height = size?.height || 100;
      break;
    default:
      throw new Error(`Unknown shape type: ${type}`);
  }

  slide.data.objects.push(element);
  return { id: element.id, type: 'shape', element };
}

function changeSlideBackground({ type, value }, slide) {
  if (type === 'color') {
    slide.data.background = value;
  } else if (type === 'gradient') {
    slide.data.background = value; // Would need more complex handling for gradients
  }
  
  return { background: { type, value } };
}

function addImageToSlide({ src, position, size }, slide) {
  const element = {
    type: 'Image',
    left: position?.x || 100,
    top: position?.y || 100,
    src: src,
    scaleX: size?.width ? size.width / 100 : 1,
    scaleY: size?.height ? size.height / 100 : 1,
    id: Date.now()
  };

  slide.data.objects.push(element);
  return { id: element.id, type: 'image', element };
}

function modifySlideElement({ elementId, properties }, slide) {
  const elementIndex = slide.data.objects.findIndex(obj => obj.id == elementId);
  if (elementIndex === -1) {
    throw new Error(`Element with id ${elementId} not found`);
  }

  const element = slide.data.objects[elementIndex];
  
  // Apply properties
  Object.keys(properties).forEach(key => {
    if (key === 'position') {
      element.left = properties.position.x;
      element.top = properties.position.y;
    } else if (key === 'size') {
      if (element.type === 'IText') {
        element.fontSize = properties.size.fontSize;
      } else {
        element.width = properties.size.width;
        element.height = properties.size.height;
      }
    } else {
      element[key] = properties[key];
    }
  });

  return { id: elementId, modified: true, element };
}

function deleteSlideElement({ elementId }, slide) {
  const elementIndex = slide.data.objects.findIndex(obj => obj.id == elementId);
  if (elementIndex === -1) {
    throw new Error(`Element with id ${elementId} not found`);
  }

  slide.data.objects.splice(elementIndex, 1);
  return { id: elementId, deleted: true };
}

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

app.listen(PORT, () => {
  console.log(`🚀 Mycelium Slide API server running on port ${PORT}`);
  console.log(`📊 API Documentation: http://localhost:${PORT}/api/health`);
});