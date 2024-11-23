import React, { useState } from 'react';
import { TextField, Button, Container, Typography } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const Home = () => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false); 
  const [error, setError] = useState(null); 
  const navigate = useNavigate();

  const handleSearch = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('http://localhost:8000/search', {
        query_text: query,
        num_results_to_print: 20, 
        top_k: 300,
      });
      navigate('/articles', { state: { results: response.data } });
    } catch (error) {
      setError('Error fetching search results');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container>
      <Typography variant="h4" style={{ marginBottom: '20px', textAlign: 'center' }}>
        What article topics are you interested in?
      </Typography>

      <TextField
        label="Search for articles"
        variant="outlined"
        fullWidth
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        style={{ marginBottom: '20px' }}
      />

      <Button 
        variant="contained" 
        color="primary" 
        onClick={handleSearch}
        disabled={loading} 
      >
        {loading ? 'Searching...' : 'Search'}
      </Button>

      {error && (
        <Typography color="error" style={{ marginTop: '20px' }}>
          {error}
        </Typography>
      )}
    </Container>
  );
};

export default Home;