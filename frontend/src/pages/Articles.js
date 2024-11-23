import React, { useState } from 'react';
import { useLocation } from 'react-router-dom';
import { Container, Typography, Card, CardContent, Collapse, IconButton } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

const Articles = () => {
  const location = useLocation();
  const { results } = location.state || { results: [] };

  return (
    <Container>
      <Typography variant="h4" style={{ marginBottom: '20px', textAlign: 'center' }}>
        Search Results
      </Typography>

      {results.length === 0 ? (
        <Typography>No articles found.</Typography>
      ) : (
        results.map((article, index) => (
          <Card key={index} style={{ marginBottom: '20px' }}>
            <CardContent>
              <Typography variant="h6">{article.title}</Typography>
              <AbstractCard article={article} />
              <Typography variant="body2" color="textSecondary">
                Categories: {article.cat_text}
              </Typography>
              <a href={article.link_to_pdf} target="_blank" rel="noopener noreferrer">
                View PDF
              </a>
            </CardContent>
          </Card>
        ))
      )}
    </Container>
  );
};

const AbstractCard = ({ article }) => {
  const [expanded, setExpanded] = useState(false);

  const handleExpandClick = () => {
    setExpanded(!expanded);
  };

  return (
    <div>
      <Typography variant="body1">
        {expanded ? article.abstract : `${article.abstract.substring(0, 200)}...`}
      </Typography>
      {article.abstract.length > 200 && (
        <IconButton
          onClick={handleExpandClick}
          style={{ padding: 0 }}
        >
          <ExpandMoreIcon />
        </IconButton>
      )}
      <Collapse in={expanded} timeout="auto" unmountOnExit>
        <Typography variant="body1">{article.abstract}</Typography>
      </Collapse>
    </div>
  );
};

export default Articles;