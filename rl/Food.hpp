#pragma once

#include <SFML/Graphics.hpp>


class Food {
   private:
    sf::CircleShape shape;
    const sf::Color COLOR = sf::Color(0, 127, 127);
    const double SIZE = 50;
    sf::Vector2f mvmntVec;

   public:
    Food(sf::Vector2f &, sf::Vector2f &);
    void draw(sf::RenderWindow &);
    void move(sf::RenderWindow &);
    sf::Vector2f getPos();
};