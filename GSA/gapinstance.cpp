#include "gapinstance.hpp"

#include <fstream>
#include <iomanip>

using namespace stdgap;

std::ostream& GapInstance::format(
        std::ostream& os,
        int verbosity_level) const
{
    if (verbosity_level >= 1) {
        os
            << "Number of agents:  " << number_of_agents() << std::endl
            << "Number of items:   " << number_of_items() << std::endl
            << "Total cost:        " << total_cost() << std::endl
            << "Maximum cost:      " << maximum_cost() << std::endl
            << "Maximum weight:    " << maximum_weight() << std::endl
            ;
    }

    if (verbosity_level >= 2) {
        os
            << std::endl
            << std::setw(12) << "Agent"
            << std::setw(12) << "Capacity"
            << std::endl
            << std::setw(12) << "-----"
            << std::setw(12) << "--------"
            << std::endl;
        for (AgentIdx agent_id = 0;
                agent_id < number_of_agents();
                ++agent_id) {
            os
                << std::setw(12) << agent_id
                << std::setw(12) << capacity(agent_id)
                << std::endl;
        }

        os
            << std::endl
            << std::setw(12) << "Item"
            << std::setw(12) << "Agent"
            << std::setw(12) << "Weight"
            << std::setw(12) << "Cost"
            << std::endl
            << std::setw(12) << "----"
            << std::setw(12) << "-----"
            << std::setw(12) << "------"
            << std::setw(12) << "----"
            << std::endl;
        for (ItemIdx item_id = 0;
                item_id < number_of_items();
                ++item_id) {
            for (AgentIdx agent_id = 0;
                    agent_id < number_of_agents();
                    ++agent_id) {
                os
                    << std::setw(12) << item_id
                    << std::setw(12) << agent_id
                    << std::setw(12) << weight(item_id, agent_id)
                    << std::setw(12) << cost(item_id, agent_id)
                    << std::endl;
            }
        }
    }

    return os;
}

void GapInstance::write(std::string instance_path)
{
    std::ofstream file(instance_path);
    if (!file.good()) {
        throw std::runtime_error(
                "Unable to open file \"" + instance_path + "\".");
    }

    file << number_of_agents() << " " << number_of_items() << std::endl;
    for (AgentIdx agent_id = 0; agent_id < number_of_agents(); ++agent_id) {
        for (ItemIdx item_id = 0; item_id < number_of_items(); ++item_id)
            file << item(item_id).alternatives[agent_id].cost << " ";
        file << std::endl;
    }
    for (AgentIdx agent_id = 0; agent_id < number_of_agents(); ++agent_id) {
        for (ItemIdx item_id = 0; item_id < number_of_items(); ++item_id)
            file << item(item_id).alternatives[agent_id].weight << " ";
        file << std::endl;
    }
    for (AgentIdx agent_id = 0; agent_id < number_of_agents(); ++agent_id)
        file << capacity(agent_id) << " ";
    file << std::endl;
    file.close();
}

void GapInstance::add_agents(AgentIdx number_of_agents)
{
    this->capacities_.insert(this->capacities_.end(), number_of_agents, 0);
    for (ItemIdx item_id = 0; item_id < this->number_of_items(); ++item_id) {
        this->items_[item_id].alternatives.insert(
                this->items_[item_id].alternatives.end(),
                number_of_agents,
                Alternative());
    }
}

void GapInstance::set_capacity(
        AgentIdx agent_id,
        Weight capacity)
{
    this->capacities_[agent_id] = capacity;
}

void GapInstance::add_items(ItemIdx number_of_items)
{
    Item item;
    item.alternatives.insert(
            item.alternatives.end(),
            this->number_of_agents(),
            Alternative());
    this->items_.insert(
            this->items_.end(),
            number_of_items,
            item);
}

void GapInstance::set_weight(
        ItemIdx item_id,
        AgentIdx agent_id,
        Weight weight)
{
    this->items_[item_id].alternatives[agent_id].weight = weight;
}

void GapInstance::set_cost(
        ItemIdx item_id,
        AgentIdx agent_id,
        Cost cost)
{
    this->items_[item_id].alternatives[agent_id].cost = cost;
}

void GapInstance::read(
        const std::string& instance_path,
        const std::string& format)
{
    std::ifstream file(instance_path);
    if (!file.good()) {
        throw std::runtime_error(
                "Unable to open file \"" + instance_path + "\".");
    }

    if (format == "orlibrary" || format == "") {
        read_orlibrary(file);
    } else if (format == "standard") {
        read_standard(file);
    } else {
        throw std::invalid_argument(
                "Unknown instance format \"" + format + "\".");
    }

    file.close();
}

void GapInstance::read_orlibrary(
        std::ifstream& file)
{
    ItemIdx number_of_items;
    AgentIdx number_of_agents;
    file >> number_of_agents >> number_of_items;

    add_agents(number_of_agents);
    add_items(number_of_items);

    Cost cost;
    for (AgentIdx agent_id = 0; agent_id < number_of_agents; ++agent_id) {
        for (ItemPos item_id = 0; item_id < number_of_items; ++item_id) {
            file >> cost;
            set_cost(
                    item_id,
                    agent_id,
                    cost);
        }
    }

    Cost weight;
    for (AgentIdx agent_id = 0; agent_id < number_of_agents; ++agent_id) {
        for (ItemPos item_id = 0; item_id < number_of_items; ++item_id) {
            file >> weight;
            set_weight(
                    item_id,
                    agent_id,
                    weight);
        }
    }

    Weight capacity = -1;
    for (AgentIdx agent_id = 0; agent_id < number_of_agents; ++agent_id) {
        file >> capacity;
        set_capacity(
                agent_id,
                capacity);
    }
}

void GapInstance::read_standard(
        std::ifstream& file)
{
    ItemIdx number_of_items;
    AgentIdx number_of_agents;
    file >> number_of_agents >> number_of_items;

    add_agents(number_of_agents);
    add_items(number_of_items);

    Weight capacity = -1;
    for (AgentIdx agent_id = 0; agent_id < number_of_agents; ++agent_id) {
        file >> capacity;
        set_capacity(
                agent_id,
                capacity);
    }

    Weight weight;
    Cost cost;
    for (ItemPos item_id = 0; item_id < number_of_items; ++item_id) {
        for (AgentIdx agent_id = 0; agent_id < number_of_agents; ++agent_id) {
            file >> weight >> cost;
            set_weight(
                    item_id,
                    agent_id,
                    weight);
            set_cost(
                    item_id,
                    agent_id,
                    cost);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// Build /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void GapInstance::build()
{
    // Compute total and maximum weight and cost of the instance.
    for (ItemIdx item_id = 0;
            item_id < this->number_of_items();
            ++item_id) {

        for (AgentIdx agent_id = 0;
                agent_id < this->number_of_agents();
                ++agent_id) {
            Cost cost = this->cost(item_id, agent_id);
            Cost weight = this->weight(item_id, agent_id);

            this->total_cost_ += cost;
            this->maximum_cost_ = std::max(this->maximum_cost_, cost);
            this->maximum_weight_ = std::max(this->maximum_weight_, weight);
            this->items_[item_id].total_weight += weight;
            this->items_[item_id].total_cost += cost;

            if (this->items_[item_id].maximum_weight_agent_id == -1
                    || this->items_[item_id].maximum_weight < weight) {
                this->items_[item_id].maximum_weight = weight;
                this->items_[item_id].maximum_weight_agent_id = agent_id;
            }

            if (this->items_[item_id].minimum_weight_agent_id == -1
                    || this->items_[item_id].minimum_weight > weight) {
                this->items_[item_id].minimum_weight = weight;
                this->items_[item_id].minimum_weight_agent_id = agent_id;
            }

            if (this->items_[item_id].maximum_cost_agent_id == -1
                    || this->items_[item_id].maximum_cost < cost) {
                this->items_[item_id].maximum_cost = cost;
                this->items_[item_id].maximum_cost_agent_id = agent_id;
            }

            if (this->items_[item_id].minimum_cost_agent_id == -1
                    || this->items_[item_id].minimum_cost > cost) {
                this->items_[item_id].minimum_cost = cost;
                this->items_[item_id].minimum_cost_agent_id = agent_id;
            }
        }

        this->sum_of_minimum_costs_ += this->item(item_id).minimum_cost;
    }

    //return std::move(instance_);
}