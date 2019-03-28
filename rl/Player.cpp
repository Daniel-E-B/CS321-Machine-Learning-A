#include "Player.hpp"
#include <SFML/Graphics.hpp>

Player::Player(sf::Vector2f &pos) {
    shape = sf::CircleShape(SIZE);
    shape.setFillColor(COLOR);
    shape.setOrigin(SIZE, SIZE);
    shape.setPosition(pos);
}

void Player::draw(sf::RenderWindow &window) {
    window.draw(shape);
}

void Player::move(sf::Vector2f &offset) {
    shape.move(offset);
}