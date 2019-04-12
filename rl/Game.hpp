#pragma once

#include <vector>

#include <SFML/Graphics.hpp>

#include "Basket.hpp"
#include "Food.hpp"
#include "tiny-dnn/tiny_dnn/tiny_dnn.h"

class Game {
   private:
    Food *food;
    Basket *basket;
    const int MAX_SPEED_COMPONENT = 5;
    const int TICKS_PER_GENERATION = 150/*0*/; // TODO: tune
    const int CREATURES = 200;
    const double MUTATION_RATE = 1; // % chance of mutation
    const double MUTATION_SIZE = 0.2; // max % size of mutation
    std::vector<double> fitnesses;
    std::vector<std::string> brains;
    double fitness();  // ideal is 0
    void tick(sf::RenderWindow &);
    void mutate();
    void reset(sf::RenderWindow &);

   public:
    void draw(sf::RenderWindow &);
    Game(sf::RenderWindow &);                                             
    void generation(sf::RenderWindow &, unsigned long long int, bool &);  // TODO: high level interface to go through each basket with a randomly generated food for n ticks, then mutate all
};