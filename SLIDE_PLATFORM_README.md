# Mycelium Slide Platform

A comprehensive slide-making platform with LLM integration, built on top of the Mycelium federated learning framework.

![Platform Overview](https://github.com/user-attachments/assets/b4736dbe-b6dc-4e7b-8d17-051008b95d96)

## 🚀 Features

### Slide Editor
- **Canvas-based editing** with drag-and-drop functionality
- **Multi-slide management** with thumbnails and navigation
- **Rich element support**: Text, shapes, images, charts
- **Real-time preview** and editing
- **Keyboard shortcuts** for power users

![Slide Editor Interface](https://github.com/user-attachments/assets/99e5f9da-c3a3-4b40-be3c-d4adb168121b)

### LLM Integration
- **JSON-based command interface** for programmatic slide manipulation
- **Natural language processing** for intuitive commands
- **Batch operations** for complex slide modifications
- **RESTful API** for external integrations
- **Tool calling interface** compatible with major LLM providers

### Production Ready Features
- **Express.js backend** with robust API
- **Canvas manipulation** using Fabric.js
- **Responsive design** with Tailwind CSS
- **Hot module reloading** for development
- **Modular architecture** for easy extension

## 🏗️ Architecture

```
mycelium/
├── frontend/                 # React + Vite frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── SlideEditor.jsx     # Main slide editor
│   │   │   └── SlideLLMAPI.js      # LLM integration
│   │   ├── App.jsx           # Main app component
│   │   └── hero.jsx          # Landing page
│   └── package.json
├── api/                      # Express.js backend
│   ├── server.js             # Main API server
│   └── package.json
└── README.md
```

## 🛠️ Installation & Setup

### Prerequisites
- Node.js (v16+)
- npm or yarn

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Backend Setup
```bash
cd api
npm install
npm start
```

### Development
```bash
# Terminal 1 - Frontend
cd frontend && npm run dev

# Terminal 2 - Backend  
cd api && npm run dev
```

## 📚 LLM Integration Guide

### JSON Command Structure

All LLM commands follow this structure:
```json
{
  "action": "command_name",
  "parameter1": "value1",
  "parameter2": "value2"
}
```

### Available Commands

#### Add Text
```json
{
  "action": "add_text",
  "content": "Hello World",
  "position": { "x": 150, "y": 100 },
  "style": {
    "fontSize": 24,
    "color": "#333333",
    "fontFamily": "Arial",
    "bold": false,
    "italic": false
  }
}
```

#### Add Shapes
```json
{
  "action": "add_shape",
  "type": "rectangle",
  "position": { "x": 200, "y": 200 },
  "size": { "width": 150, "height": 100 },
  "style": {
    "fill": "#3b82f6",
    "stroke": "#1e40af",
    "strokeWidth": 2
  }
}
```

#### Change Background
```json
{
  "action": "change_background",
  "type": "color",
  "value": "#f8fafc"
}
```

#### Add Image
```json
{
  "action": "add_image",
  "src": "https://example.com/image.jpg",
  "position": { "x": 100, "y": 100 },
  "size": { "width": 200, "height": 150 }
}
```

#### Modify Element
```json
{
  "action": "modify_element",
  "elementId": "1234567890",
  "properties": {
    "position": { "x": 300, "y": 200 },
    "size": { "width": 200, "height": 100 }
  }
}
```

### Natural Language Support

The platform also supports natural language commands:
- `"add text Hello World"`
- `"add rectangle"`
- `"change background #ff0000"`
- `"add circle"`

### API Endpoints

#### Presentations
- `GET /api/presentations` - List all presentations
- `POST /api/presentations` - Create new presentation
- `GET /api/presentations/:id` - Get specific presentation
- `PUT /api/presentations/:id` - Update presentation
- `DELETE /api/presentations/:id` - Delete presentation

#### LLM Integration
- `POST /api/presentations/:id/slides/:slideId/llm-command` - Execute single command
- `POST /api/presentations/:id/llm-batch` - Execute batch commands

#### Templates
- `GET /api/templates` - Get available templates

### Tool Calling Interface

For LLM providers that support tool calling:

```javascript
const tools = [
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
        }
      },
      required: ['content']
    }
  }
];
```

## 🎯 Use Cases

### 1. Automated Presentation Generation
```javascript
// Generate a presentation from data
const commands = [
  { action: 'add_text', content: 'Q4 Results', position: { x: 300, y: 50 } },
  { action: 'add_shape', type: 'rectangle', position: { x: 100, y: 150 } },
  { action: 'change_background', type: 'color', value: '#f0f9ff' }
];

await api.batchExecute(commands);
```

### 2. AI-Powered Design Assistant
```javascript
// LLM can analyze content and suggest layouts
const suggestion = await llm.generateSlideLayout(content);
await api.executeLLMCommand(suggestion);
```

### 3. Bulk Slide Operations
```javascript
// Apply consistent styling across multiple slides
const styleCommands = slides.map(slide => ({
  slideId: slide.id,
  command: { action: 'change_background', type: 'color', value: '#ffffff' }
}));

await api.batchExecute(styleCommands);
```

## 🔧 Customization

### Adding New Element Types
1. Extend the `SlideLLMAPI.js` class
2. Add new command handlers in `server.js`
3. Update the Fabric.js canvas implementation

### Custom Templates
```javascript
// Add new template
templates['corporate'] = {
  id: 'corporate',
  name: 'Corporate Template',
  slides: [
    {
      data: {
        objects: [/* template elements */],
        background: '#1e40af'
      }
    }
  ]
};
```

## 📈 Performance Considerations

- **Canvas optimization**: Use object caching for large presentations
- **Memory management**: Implement slide virtualization for 100+ slides
- **API rate limiting**: Add rate limiting for LLM commands
- **Image optimization**: Implement lazy loading for images

## 🔄 Integration Examples

### OpenAI GPT Integration
```javascript
const response = await openai.chat.completions.create({
  model: "gpt-4",
  messages: [{ role: "user", content: "Create a slide about AI trends" }],
  tools: slidePlatformTools
});

if (response.choices[0].message.tool_calls) {
  await api.executeCommand(response.choices[0].message.tool_calls[0].function);
}
```

### Claude Integration
```javascript
const command = await anthropic.messages.create({
  model: "claude-3-sonnet-20240229",
  messages: [{ role: "user", content: "Add a title slide" }],
  tools: slidePlatformTools
});

await api.executeCommand(command.content[0].input);
```

## 🚀 Future Enhancements

- [ ] Real-time collaboration support
- [ ] Animation and transition effects
- [ ] Export to PowerPoint/PDF
- [ ] Template marketplace
- [ ] Advanced chart/diagram library
- [ ] Voice-to-slide generation
- [ ] Mobile app support
- [ ] Cloud storage integration

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📧 Contact

- **Ujjwal**: [LinkedIn](https://www.linkedin.com/in/ujjwalsinghh/)
- **Jon**: [LinkedIn](https://www.linkedin.com/in/jkozlik/)
- **Sriya**: [LinkedIn](https://www.linkedin.com/in/sriya-chinthalapudi/)

---

Built with ❤️ using React, Express.js, Fabric.js, and modern web technologies.