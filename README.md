# College Recommendation System

A machine learning-based college recommendation system that helps students find the best colleges based on their academic performance and preferred branch of study.

## Features

- **Smart Recommendations**: Uses ML model trained on historical cutoff data and placement statistics
- **Multiple Branches**: Supports various engineering branches (CS, Electronics, Mechanical, etc.)
- **Real-time Predictions**: Get instant recommendations based on your percentage
- **Beautiful UI**: Modern, responsive web interface
- **Scalable**: Easy to add new colleges and update data

## How It Works

1. **Data Collection**: The system uses historical data including:
   - 3-year cutoff percentages for each branch
   - Placement statistics
   - College rankings
   - Average package information

2. **ML Model**: A Random Forest Regressor that:
   - Analyzes patterns in cutoff trends
   - Considers placement success rates
   - Factors in college reputation
   - Predicts best-fit colleges

3. **Recommendation Engine**: Provides personalized recommendations based on:
   - Student's percentage
   - Preferred branch
   - Historical data patterns

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd college-recommendation-system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Open your browser** and go to `http://localhost:5000`

## Usage

1. **Enter Your Details**:
   - Input your percentage (0-100%)
   - Select your preferred branch
   - Choose number of recommendations (3, 5, or 10)

2. **Get Recommendations**:
   - Click "Get Recommendations"
   - View your personalized college list
   - Each recommendation includes a confidence score

## API Endpoints

### POST `/api/recommend`
Get college recommendations for a student.

**Request Body**:
```json
{
    "percentage": 95.5,
    "branch": "Computer Science",
    "top_n": 5
}
```

**Response**:
```json
{
    "student_percentage": 95.5,
    "branch": "Computer Science",
    "recommendations": [
        {
            "rank": 1,
            "college": "IIT Bombay",
            "score": 0.9234
        }
    ]
}
```

### GET `/api/branches`
Get list of available branches.

### GET `/api/colleges`
Get list of available colleges.

### POST `/api/train`
Retrain the model with updated data.

## Data Structure

The system expects data in the following format:

```csv
college,branch,year,cutoff_2021,cutoff_2022,cutoff_2023,placement_percentage,avg_package_lpa,ranking
IIT Bombay,Computer Science,2023,98.5,98.2,98.8,95.5,18.5,1
```

## Customization

### Adding New Colleges
1. Update the `create_sample_data()` method in `data_processor.py`
2. Add college information to the colleges list
3. Retrain the model using `/api/train`

### Adding New Branches
1. Update the branches list in `data_processor.py`
2. Add corresponding data for the new branch
3. Retrain the model

### Updating Data
1. Replace the sample data with real data
2. Ensure data follows the required format
3. Retrain the model

## Model Performance

The system uses a Random Forest Regressor with the following features:
- Branch encoding
- Average cutoff percentage
- Cutoff trend analysis
- Placement percentage
- Average package
- College ranking

## Future Enhancements

- [ ] Real-time data integration
- [ ] More sophisticated ML algorithms
- [ ] User feedback integration
- [ ] Advanced filtering options
- [ ] College comparison features
- [ ] Mobile app development

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For support or questions, please open an issue in the repository.

