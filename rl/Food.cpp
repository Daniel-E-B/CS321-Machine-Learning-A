#include <SFML/Graphics.hpp>

#include "Food.hpp"

Food::Food(sf::Vector2f &pos) {
    shape = sf::CircleShape(SIZE);
    shape.setFillColor(COLOR);
    shape.setOrigin(SIZE, SIZE);
    shape.setPosition(pos);
}

void Food::draw(sf::RenderWindow &window) {
    window.draw(shape);
}

void Food::move(sf::Vector2f &offset) {
    shape.move(offset);
}