#pragma once

#include <vector>

#include <SFML/Graphics.hpp>

#include "Basket.hpp"
#include "Food.hpp"

class Game {
   private:
    Food *food;
    Basket *basket;
    int currentFood = 0;
    const int MAX_SPEED_COMPONENT = 5;
    double fitness(); // ideal is 0
    /*
    TODO: const paramters for:
        * ticks / gen
        * mutation quantity & rate
     */

   public:
    void draw(sf::RenderWindow &);
    Game(sf::RenderWindow &);       // num foods
    void tick(sf::RenderWindow &);  // TODO: private
    void mutate();                  // TODO: private
    void generation();              // TODO: high level interface to go through each basket with a randomly generated food for n ticks, then mutate all
};