import express from 'express';
import open from 'open';
import fs from 'fs';

const app = express();
const PORT = 3000;

app.post('/run',(req,res)=>{
    const command = 'cd ../Console_App && conda activate ml && python main.py';
    const scriptContent = `start cmd /k "${command}"`;
  fs.writeFileSync('runScripts.bat', scriptContent);
  open('runScripts.bat', { wait: false });
})

    app.listen(PORT, () => {
        console.log(`Server running on port ${PORT}`);
    });