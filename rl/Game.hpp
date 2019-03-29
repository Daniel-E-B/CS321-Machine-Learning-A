#pragma once

#include <vector>

#include <SFML/Graphics.hpp>

#include "Basket.hpp"
#include "Drawable.hpp"
#include "Food.hpp"

class Game : public Drawable {
   private:
    Food *food;
    Basket *basket;
    int currentFood = 0;

   public:
    void draw(sf::RenderWindow &);
    Game(sf::RenderWindow &);  // num foods
    bool tick();                    // returns true if generation is complete
    void mutate();                  // probably have parameters
    void setFood(sf::Vector2f &, sf::Vector2f &);              // pos, mvmntvec
    double getFitness();
};