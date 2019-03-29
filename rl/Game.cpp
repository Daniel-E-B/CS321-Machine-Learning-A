#include <cstdlib>

#include <SFML/Graphics.hpp>

#include "Basket.hpp"
#include "Food.hpp"
#include "Game.hpp"

Game::Game(sf::RenderWindow &window) {
    food = new Food(*(new sf::Vector2f(window.getSize().x / 2, window.getSize().y / 2)), *(new sf::Vector2f((std::rand() % 100) / 20.0, (std::rand() % 100) / 20.0)));
    basket = new Basket(*(new sf::Vector2f(window.getSize().x / 2, window.getSize().y)));
}

void Game::draw(sf::RenderWindow &window) {
    basket->draw(window);
    food->draw(window);
}

void Game::setFood(sf::Vector2f &pos, sf::Vector2f &mvmntVec) {
}

bool Game::tick() {
    food->move();
}